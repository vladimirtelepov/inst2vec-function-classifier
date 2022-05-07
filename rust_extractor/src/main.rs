/// arswitch_readphy_internal|int,int,int,int|Access,PHYs,integrated,into,the,switch,chip,through,the,switch,s,MDIO,control,register
use proc_macro2::LineColumn;
use quote::quote;
use serde::Serialize;

use std::env;
use std::fs::File;
use std::io::Read;
use std::process;
use syn::spanned::Spanned;

fn main() {
    let mut args = env::args();
    let _ = args.next(); // executable name

    let (in_filename, out_filename) = match (args.next(), args.next()) {
        (Some(in_filename), Some(out_filename)) => (in_filename, out_filename),
        _ => {
            eprintln!("Usage: rust_extractor <in_filename> <out_filename>");
            process::exit(1);
        }
    };

    let mut file = File::open(&in_filename).expect("Unable to open file");

    let mut src = String::new();
    file.read_to_string(&mut src).expect("Unable to read file");

    let syntax = syn::parse_file(&src).expect("Unable to parse file");

    let input = std::fs::read_to_string(in_filename).unwrap();
    let fn_infos = extract_info(&input, &syntax.items);

    let json = serde_json::to_string(&fn_infos).unwrap();
    std::fs::write(out_filename, json).expect("failed to write to the file");
}

#[derive(Debug, Serialize)]
struct ExtractedInfo {
    pub name: String,
    pub arg_types: Vec<String>,
    pub output_type: String,
    pub comments: Vec<String>,
    pub func: String,
}

fn extract_info(input: &str, items: &[syn::Item]) -> Vec<ExtractedInfo> {
    let mut result = Vec::new();
    let mut last_item_end = LineColumn { line: 0, column: 0 };

    for item in items {
        match item {
            syn::Item::Fn(fn_item) => {
                result.push(extract_info_impl(
                    input,
                    &fn_item.attrs,
                    &fn_item.sig,
                    last_item_end,
                    fn_item.span().start(),
                    quote!(#fn_item).to_string(),
                ));
                last_item_end = item.span().end();
            }

            syn::Item::Impl(impl_item) => {
                last_item_end = impl_item.brace_token.span.end();

                for item in &impl_item.items {
                    match item {
                        syn::ImplItem::Method(method_item)
                            if !method_item.block.stmts.is_empty() =>
                        {
                            result.push(extract_info_impl(
                                input,
                                &method_item.attrs,
                                &method_item.sig,
                                last_item_end,
                                method_item.span().start(),
                                quote!(#method_item).to_string(),
                            ));

                            last_item_end = item.span().end();
                        }

                        i => {
                            last_item_end = i.span().end();
                        }
                    }
                }

                last_item_end = impl_item.span().end();
            }
            item => {
                last_item_end = item.span().end();
            }
        }
    }

    result
}

fn extract_info_impl(
    input: &str,
    attrs: &[syn::Attribute],
    sig: &syn::Signature,
    last_item_end: LineColumn,
    current_item_start: LineColumn,
    func: String,
) -> ExtractedInfo {
    let mut comments = extracts_comments_from_attrs(attrs);
    if last_item_end < current_item_start {
        let comment = extract_comments(input, last_item_end, current_item_start);
        if !comment.is_empty() && !comment.chars().all(|c| c == ',') {
            comments.push(comment);
        }
    }

    let output_type = extract_output_type(&sig.output);
    let name = sig.ident.to_string();
    let arg_types = extract_arguments(sig);

    ExtractedInfo {
        name,
        arg_types,
        output_type,
        comments,
        func,
    }
}

fn extract_comments(
    input: &str,
    last_item_end: LineColumn,
    current_item_start: LineColumn,
) -> String {
    let mut last_item_offset = 0;
    for _ in 1..last_item_end.line {
        let newline_pos = match input[last_item_offset..].find("\n") {
            Some(pos) => pos,
            None => return String::new(),
        };
        last_item_offset += newline_pos + 1;
    }
    last_item_offset += last_item_end.column;

    let mut current_item_offset = 0;
    for _ in 1..current_item_start.line {
        let newline_pos = match input[current_item_offset..].find("\n") {
            Some(pos) => pos,
            None => return String::new(),
        };
        current_item_offset += newline_pos + 1;
    }
    current_item_offset += current_item_start.column;

    if current_item_offset <= last_item_offset || current_item_offset >= input.len() {
        return String::new();
    }

    let t = input[last_item_offset..current_item_offset].to_string();
    let mut t = t.replace("\n", ",");
    t.retain(|c| (c != '/'));

    t.trim_start_matches(',')
        .trim_start_matches(' ')
        .trim_end_matches(',')
        .trim_end_matches(' ')
        .to_string()
}

fn extracts_comments_from_attrs(attrs: &[syn::Attribute]) -> Vec<String> {
    let mut result = Vec::new();

    for attr in attrs {
        for segment in attr.path.segments.iter() {
            let segment_as_str = format!("{}", segment.ident);
            if segment_as_str == "doc" {
                let mut attr_as_str = format!("{}", attr.tokens).to_string();

                const START_SEQ: &str = r#"= ""#;
                const END_SEQ: &str = r#"""#;
                if attr_as_str.starts_with(START_SEQ) {
                    attr_as_str = attr_as_str[START_SEQ.len()..].to_string();
                }

                if attr_as_str.ends_with(END_SEQ) {
                    attr_as_str = attr_as_str[..attr_as_str.len() - END_SEQ.len()].to_string();
                }

                result.push(attr_as_str.to_string());
            }
        }
    }

    result
}

fn extract_output_type(output_type: &syn::ReturnType) -> String {
    match output_type {
        syn::ReturnType::Default => "void".to_string(),
        syn::ReturnType::Type(_, ty) => quote!(-> #ty).to_string(),
    }
}

fn extract_arguments(signature: &syn::Signature) -> Vec<String> {
    let mut result = Vec::new();
    for input in &signature.inputs {
        match input {
            syn::FnArg::Typed(typed) => {
                let ty = &typed.ty;
                result.push(quote!(#ty).to_string());
            }
            _ => {
                // debug!("self shouldn't be in singular functions");
            }
        }
    }

    result
}
