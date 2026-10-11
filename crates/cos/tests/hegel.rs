//! Property-based tests with Hegel (a pilot next to the proptest ones in `writer.rs`): object
//! serialization, full saves, editing sequences checked against a model, encryption, and saving
//! whatever the reader accepts from a damaged file.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::sync::Arc;

use hegel::TestCase;
use hegel::generators::{self as gs, Generator};
use pdfcraft_cos::{
    Algorithm, Auth, CosError, Dict, Document, Lexer, NewEncryption, ObjRef, Object, PdfString, SaveOptions, Stream, serialize, write_full,
    write_incremental,
};

/// Names readers act on, which random bytes almost never spell.
const NAMES: &[&[u8]] = &[
    b"Type",
    b"Length",
    b"Filter",
    b"FlateDecode",
    b"DecodeParms",
    b"XRef",
    b"ObjStm",
    b"N",
    b"First",
    b"Root",
    b"Prev",
    b"Size",
    b"Encrypt",
    b"ID",
    b"Pages",
    b"Kids",
    b"Parent",
    b"Catalog",
];

/// File-structure keywords and delimiters, for string and stream data.
const FRAGMENTS: &[&[u8]] = &[
    b"endstream",
    b"endobj",
    b"stream\r\n",
    b"1 0 obj",
    b"1 0 R",
    b"xref",
    b"trailer",
    b"startxref",
    b"%%EOF",
    b"\r",
    b"\n",
    b"(",
    b")",
    b"\\",
    b"<<",
    b">>",
    b"%",
];

/// Characters SASLprep (RFC 4013) maps to nothing, to a space or by NFKC, which AES-256
/// passwords go through, next to the plain characters they become.
const SASLPREP: &str = "a\u{AD}\u{200B}\u{FEFF}\u{A0} \u{FB01}fi\u{FF21}AE\u{301}\u{C9}\u{212B}\u{C5}";

const ALGORITHMS: [Algorithm; 4] = [Algorithm::Rc4_40, Algorithm::Rc4_128, Algorithm::Aes128, Algorithm::Aes256];

#[hegel::composite]
fn data(tc: &TestCase, max_chunks: usize) -> Vec<u8> {
    let chunk = gs::one_of(vec![gs::binary().max_size(6).boxed(), gs::sampled_from(FRAGMENTS).map(<[u8]>::to_vec).boxed()]);
    tc.draw_silent(gs::vecs(chunk).max_size(max_chunks)).concat()
}

#[hegel::composite]
fn name(tc: &TestCase) -> Vec<u8> {
    if tc.draw_silent(gs::booleans()) { tc.draw_silent(gs::sampled_from(NAMES)).to_vec() } else { tc.draw_silent(gs::binary().max_size(8)) }
}

#[hegel::composite]
fn dict(tc: &TestCase, depth: u32, refs: u32) -> Dict {
    let mut d = Dict::new();
    for _ in 0..tc.draw_silent(gs::integers::<u8>().max_value(4)) {
        d.set(tc.draw_silent(name()), tc.draw_silent(object(depth, refs, false)));
    }
    d
}

/// A direct object nested at most `depth` deep (a stream only at the top when `stream`).
/// References point at objects 0..=`refs`, so some dangle and some form cycles.
#[hegel::composite]
fn object(tc: &TestCase, depth: u32, refs: u32, stream: bool) -> Object {
    let kinds = match (depth, stream) {
        (0, _) => 6,
        (_, false) => 8,
        (_, true) => 9,
    };
    match tc.draw_silent(gs::integers::<u8>().max_value(kinds)) {
        0 => Object::Null,
        1 => Object::Bool(tc.draw_silent(gs::booleans())),
        2 => Object::Int(tc.draw_silent(gs::integers::<i64>())),
        3 => Object::Real(tc.draw_silent(gs::floats::<f64>())),
        4 => Object::String(PdfString { bytes: tc.draw_silent(data(4)), hex: tc.draw_silent(gs::booleans()) }),
        5 => Object::Name(tc.draw_silent(name())),
        6 => Object::Ref(ObjRef::new(tc.draw_silent(gs::integers::<u32>().max_value(refs)), 0)),
        7 => Object::Array(tc.draw_silent(gs::vecs(object(depth - 1, refs, false)).max_size(4))),
        8 => Object::Dict(tc.draw_silent(dict(depth - 1, refs))),
        _ => Object::Stream(Stream::from_raw(tc.draw_silent(dict(depth - 1, refs)), tc.draw_silent(data(6)))),
    }
}

#[hegel::composite]
fn objects(tc: &TestCase, max: usize) -> Vec<Object> {
    tc.draw_silent(gs::vecs(object(2, 12, true)).max_size(max))
}

/// A password the engine's Protect edit accepts: printable ASCII for the older algorithms,
/// any non-empty text for AES-256.
#[hegel::composite]
fn password(tc: &TestCase, algorithm: Algorithm) -> String {
    let ascii: String = (' '..='~').collect();
    let text = match algorithm {
        Algorithm::Aes256 => gs::one_of(vec![gs::text().boxed(), gs::text().alphabet(SASLPREP).boxed()]).boxed(),
        _ => gs::text().alphabet(&ascii).boxed(),
    };
    tc.draw_silent(text.filter(|p: &String| !p.is_empty()))
}

/// A document whose catalog reaches `objects` (numbered from 3, after the page tree and the
/// catalog that `Document::new_empty` makes).
fn document(objects: &[Object]) -> Document {
    let mut doc = Document::new_empty();
    let refs = objects.iter().map(|o| Object::Ref(doc.add(o.clone()))).collect();
    let root = doc.root().expect("a catalog");
    doc.update_dict(root, |d| d.set(b"Extra".to_vec(), Object::Array(refs))).expect("the catalog is a dictionary");
    doc
}

fn number(o: &Object) -> Option<f64> {
    match o {
        Object::Int(i) => Some(*i as f64),
        Object::Real(r) => Some(if r.is_finite() { *r } else { 0.0 }),
        _ => None,
    }
}

fn without_length(d: &Dict) -> Dict {
    d.iter().filter(|(k, _)| k.as_slice() != b"Length").map(|(k, v)| (k.clone(), v.clone())).collect()
}

/// Whether `b` is what reading `a` back should give. The writer prints integral reals as
/// integers and non-finite ones as 0, chooses literal or hex string syntax itself, and sets
/// each stream's `/Length`.
fn same(a: &Object, b: &Object) -> bool {
    match (a, b) {
        (Object::Int(x), Object::Int(y)) => x == y,
        (Object::Int(_) | Object::Real(_), Object::Int(_) | Object::Real(_)) => number(a) == number(b),
        (Object::String(x), Object::String(y)) => x.bytes == y.bytes,
        (Object::Array(x), Object::Array(y)) => x.len() == y.len() && x.iter().zip(y).all(|(x, y)| same(x, y)),
        (Object::Dict(x), Object::Dict(y)) => same_dict(x, y),
        (Object::Stream(x), Object::Stream(y)) => same_dict(&without_length(&x.dict), &without_length(&y.dict)) && *x.raw == *y.raw,
        _ => a == b,
    }
}

fn same_dict(x: &Dict, y: &Dict) -> bool {
    x.len() == y.len() && x.iter().zip(y.iter()).all(|((kx, vx), (ky, vy))| kx == ky && same(vx, vy))
}

/// Walks two documents in step from corresponding objects, following references on both sides
/// (a renumbering save changes the numbers, not what they point at).
struct Walk<'a> {
    a: &'a Document,
    b: &'a Document,
    seen: HashSet<(Option<u32>, Option<u32>)>,
}

impl<'a> Walk<'a> {
    fn roots(a: &'a Document, b: &'a Document) {
        let mut walk = Walk { a, b, seen: HashSet::new() };
        let (ra, rb) = (a.root().expect("a catalog"), b.root().expect("a catalog"));
        if let Err(e) = walk.compare(&Object::Ref(ra), &Object::Ref(rb), "Root") {
            panic!("{e}");
        }
    }

    fn compare(&mut self, x: &Object, y: &Object, path: &str) -> Result<(), String> {
        if matches!(x, Object::Ref(_)) || matches!(y, Object::Ref(_)) {
            if !self.seen.insert((x.as_ref().map(|r| r.num), y.as_ref().map(|r| r.num))) {
                return Ok(());
            }
            return self.compare(&self.a.resolve(x), &self.b.resolve(y), path);
        }
        match (x, y) {
            (Object::Array(xs), Object::Array(ys)) if xs.len() == ys.len() => {
                xs.iter().zip(ys).enumerate().try_for_each(|(i, (x, y))| self.compare(x, y, &format!("{path}[{i}]")))
            }
            (Object::Dict(xd), Object::Dict(yd)) => self.compare_dicts(xd, yd, path),
            (Object::Stream(xs), Object::Stream(ys)) if *xs.raw == *ys.raw => {
                self.compare_dicts(&without_length(&xs.dict), &without_length(&ys.dict), path)
            }
            _ if same(x, y) => Ok(()),
            _ => Err(format!("{path}: {x:?} reads back as {y:?}")),
        }
    }

    fn compare_dicts(&mut self, x: &Dict, y: &Dict, path: &str) -> Result<(), String> {
        let (kx, ky): (Vec<_>, Vec<_>) = (x.iter().map(|(k, _)| k).collect(), y.iter().map(|(k, _)| k).collect());
        if kx != ky {
            return Err(format!("{path}: keys {kx:?} read back as {ky:?}"));
        }
        x.iter().zip(y.iter()).try_for_each(|((k, x), (_, y))| self.compare(x, y, &format!("{path}/{}", String::from_utf8_lossy(k))))
    }
}

#[hegel::test]
fn serialized_objects_parse_back_to_themselves(tc: TestCase) {
    let o = tc.draw(object(3, 20, false));
    let mut bytes = Vec::new();
    serialize(&o, &mut bytes);
    let back = Lexer::new(&bytes, 0).object().expect("parses");
    assert!(same(&o, &back), "wrote {}\nread {back:?}", String::from_utf8_lossy(&bytes));
    let mut again = Vec::new();
    serialize(&back, &mut again);
    assert_eq!(String::from_utf8_lossy(&bytes), String::from_utf8_lossy(&again), "writing what was read changes the bytes");
}

#[hegel::test]
fn a_full_save_reopens_to_the_same_object_graph(tc: TestCase) {
    let objects = tc.draw(objects(8));
    let object_streams = tc.draw(gs::booleans());
    let doc = document(&objects);
    let saved = write_full(&doc, &SaveOptions { object_streams, ..SaveOptions::default() }).expect("saves");
    let back = Document::open(Arc::new(saved)).expect("reopens");
    assert!(back.repair_log().is_empty(), "the saved file needed repair: {:?}", back.repair_log());
    Walk::roots(&doc, &back);
}

/// Editing and saving incrementally, checked against a map of what each object should be.
struct Editing {
    doc: Document,
    model: BTreeMap<u32, Object>,
    freed: BTreeSet<u32>,
    snapshots: Vec<(Document, BTreeMap<u32, Object>, BTreeSet<u32>)>,
}

impl Editing {
    fn pick(&self, tc: &TestCase) -> ObjRef {
        let nums: Vec<u32> = self.model.keys().copied().collect();
        tc.assume(!nums.is_empty());
        let num = tc.draw(gs::sampled_from(nums));
        ObjRef::new(num, self.doc.generation(num))
    }

    fn new_object(&self, tc: &TestCase) -> Object {
        let refs = self.model.keys().chain(&self.freed).max().copied().unwrap_or(0) + 2;
        tc.draw(object(2, refs, true))
    }
}

#[hegel::state_machine]
impl Editing {
    #[rule(weight = 4)]
    fn add(&mut self, tc: TestCase) {
        let o = self.new_object(&tc);
        let r = self.doc.add(o.clone());
        assert!(!self.model.contains_key(&r.num) && !self.freed.contains(&r.num), "add reused {r}");
        self.model.insert(r.num, o);
    }

    #[rule(weight = 2)]
    fn replace(&mut self, tc: TestCase) {
        let r = self.pick(&tc);
        let o = self.new_object(&tc);
        self.doc.set(r, o.clone());
        self.model.insert(r.num, o);
    }

    #[rule]
    fn delete(&mut self, tc: TestCase) {
        let r = self.pick(&tc);
        self.doc.free(r);
        self.model.remove(&r.num);
        self.freed.insert(r.num);
    }

    #[rule]
    fn save_incrementally(&mut self, _: TestCase) {
        let saved = write_incremental(&self.doc, &SaveOptions::default()).expect("saves");
        self.doc = Document::open(Arc::new(saved)).expect("reopens");
        assert!(self.doc.repair_log().is_empty(), "the saved file needed repair: {:?}", self.doc.repair_log());
    }

    #[rule]
    fn snapshot(&mut self, _: TestCase) {
        self.snapshots.push((self.doc.clone(), self.model.clone(), self.freed.clone()));
    }

    #[rule]
    fn undo(&mut self, tc: TestCase) {
        tc.assume(!self.snapshots.is_empty());
        if let Some((doc, model, freed)) = self.snapshots.pop() {
            (self.doc, self.model, self.freed) = (doc, model, freed);
        }
    }

    #[invariant]
    fn objects_match_the_model(&self, _: TestCase) {
        for (num, expected) in &self.model {
            let actual = self.doc.get(ObjRef::new(*num, 0));
            assert!(same(expected, &actual), "object {num}: expected {expected:?}, read {actual:?}");
        }
        for num in &self.freed {
            assert_eq!(*self.doc.get(ObjRef::new(*num, 0)), Object::Null, "freed object {num} still reads");
        }
    }
}

#[hegel::test]
fn edits_and_incremental_saves_keep_every_object(tc: TestCase) {
    let object_streams = tc.draw(gs::booleans());
    let saved = write_full(&Document::new_empty(), &SaveOptions { object_streams, ..SaveOptions::default() }).expect("saves");
    let doc = Document::open(Arc::new(saved)).expect("reopens");
    hegel::stateful::machine(Editing { doc, model: BTreeMap::new(), freed: BTreeSet::new(), snapshots: Vec::new() }).run(tc);
}

#[hegel::test]
fn a_protected_document_opens_only_with_its_passwords(tc: TestCase) {
    let algorithm = tc.draw(gs::sampled_from(&ALGORITHMS[..]));
    let user = tc.draw(password(algorithm));
    let owner = tc.draw(password(algorithm));
    // The Protect edit refuses equal passwords.
    tc.assume(user != owner);
    let encrypt_metadata = tc.draw(gs::booleans());
    let objects = tc.draw(objects(4));
    let mut doc = document(&objects);
    let params = NewEncryption { algorithm, user_password: &user, owner_password: &owner, permissions: -1, encrypt_metadata, seed: [7; 32] };
    doc.set_encryption(&params).expect("encrypts");
    let saved = Arc::new(write_full(&doc, &SaveOptions::default()).expect("saves"));
    assert_eq!(Document::open(saved.clone()).err(), Some(CosError::NeedsPassword), "opens without a password");
    let as_user = Document::open_with_password(saved.clone(), Some(&user)).expect("the open password opens it");
    assert_eq!(as_user.security().map(|s| s.auth()), Some(Auth::User), "the open password gives owner rights");
    Walk::roots(&doc, &as_user);
    let as_owner = Document::open_with_password(saved, Some(&owner)).expect("the permissions password opens it");
    assert_eq!(as_owner.security().map(|s| s.auth()), Some(Auth::Owner));
}

#[hegel::test]
fn a_damaged_file_that_opens_saves_to_a_clean_one_that_reads_the_same(tc: TestCase) {
    let objects = tc.draw(objects(6));
    let object_streams = tc.draw(gs::booleans());
    let opts = SaveOptions { object_streams, ..SaveOptions::default() };
    let mut bytes = write_full(&document(&objects), &opts).expect("saves");
    for _ in 0..tc.draw(gs::integers::<u8>().min_value(1).max_value(3)) {
        let at = tc.draw(gs::integers::<usize>().max_value(bytes.len()));
        let cut = tc.draw(gs::integers::<usize>().max_value(8)).min(bytes.len() - at);
        let insert = tc.draw(data(2));
        bytes.splice(at..at + cut, insert);
    }
    let Ok(damaged) = Document::open(Arc::new(bytes)) else { return };
    tc.note(&format!("repairs: {:?}", damaged.repair_log()));
    let Ok(saved) = write_full(&damaged, &opts) else { return };
    let back = Document::open(Arc::new(saved)).expect("our own output reopens");
    // A root that no longer looks like a catalog sends the reader looking for another one.
    let catalog = damaged.root().and_then(|r| damaged.get(r).as_dict().map(|d| d.name(b"Type") == Some(b"Catalog") || d.contains(b"Pages")));
    tc.note(&format!("damaged root is a catalog: {catalog:?}; reopened repairs: {:?}", back.repair_log()));
    if catalog == Some(true) {
        assert!(back.repair_log().is_empty(), "our own output needed repair: {:?}", back.repair_log());
    }
    if damaged.root().is_some() {
        assert!(back.root().is_some(), "the saved file lost its root");
        Walk::roots(&damaged, &back);
    }
}
