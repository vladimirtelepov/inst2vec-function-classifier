//asdasdasd
/// this is attribute.
fn foo() {}

//asdasdasd
fn foo() {}

//
/// this is attribute.
/// asdasd
/// a
///b
#[no_mangle]
fn foo_1() -> i32 { 0 }

fn foo_2<T> (arg: T, arg2: i32)
    where T: std::fmt::Display {

}

// Rust <-> C++
// i8/i16/i32/i64
// u..
// f32/f64
// usize == size_t
// String (ptr,s,c) (<- struct) == std::string
// &str (ptr, s) (<- fat pointer/slice) == std::string_view
// Vec<> == std::vector
// HashMap<> == std::unordered_map
// HashSet<> == ...
// TreeMap<> == B-tree
// () == void
// enum == enum + union
// struct == struct
// trait = abstract class
// generic ~~ template
fn t() {}

// GADT:
// Product type: [i32] * [u32] <=> and
struct A {
    a: i32,
    b: u32,
}

// Sum type: [i32] + [u32] <=> or
enum T {
    A(i32),
    B(u32)
}

struct AA {
    aa: i32,
}

impl A {
    /// this is impl
    // this is just a comment in impl
    /* asdasd */ pub fn new(a: i32) -> Self {
        let t = a;
        Self { a }
    }

    // this
    pub fn foo(&self) {
    }
}

trait B {
    ///
    fn empty_func(&self, a: i32) -> i32;

    fn non_empty_func(&self, a: i32) -> i32 {
        0
    }
}

impl B for A {
    fn empty_func(&self, a: i32) -> i32 {
        self.a + a
    }
}

impl B for AA {
    fn empty_func(&self, a: i32) -> i32 {
        self.a + a + 1
    }

    fn non_empty_func(&self, a: i32) -> i32 {
        1
    }
}

fn foo_3(b: Box<dyn B + Display>) {

}

impl std::fmt::Display for A {
    /// this is impl trait
    // this is just a comment in impl trait
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        unimplemented!()
    }
}
