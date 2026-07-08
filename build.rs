#[path = "src/magic_table.rs"]
mod magic;

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=src/magic_table.rs");
    let rook_magic = magic::generate_rook_magic_table();
    let rook_out = map_to_rust_code(rook_magic, "ROOK_MAGICS");
    let bishop_magic = magic::generate_bishop_magic_table();
    let bishop_out = map_to_rust_code(bishop_magic, "BISHOP_MAGICS");
    std::fs::write(
        "src/magics.rs",
        format!(
            "{}\n{}\n{}",
            "use crate::magic_table::Magic;", rook_out, bishop_out
        ),
    )
    .unwrap();
}

fn map_to_rust_code<const N: usize>(magics: [magic::Magic<N>; 64], name: &str) -> String {
    let mut code = String::new();
    code.push_str(&format!("pub static {}: [Magic<{}>; 64] = [\n", name, N));
    for m in &magics {
        code.push_str(&format!(
            "    Magic {{ mask: {}, number: {}, shift: {}, entries: [{}] }},\n",
            m.mask,
            m.number,
            m.shift,
            m.entries.map(|e| e.to_string()).join(", ")
        ));
    }
    code.push_str("];\n");
    code
}
