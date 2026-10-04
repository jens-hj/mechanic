# ron patch

Source: `https://github.com/ron-rs/ron`, release `v0.12.2` (`7e8ceb454ba8`).

Backports two unreleased upstream fixes, applied unchanged from their commits:

- `72134854bb7d` — Fix quadratic float parsing in `next_bytes_is_float` (#608).
  Untyped number parsing searched the whole remaining document for `..` on
  every number. Serde reaches it whenever it skips a field it does not know
  (`IgnoredAny`), so a save carrying an unknown field took minutes to open: a
  36 MB `water.ron` with a 7 MB unrecognised `grass` list from another branch
  did not load within six minutes.
- `31529b8b8d8c` — Fix quadratic escaped string parsing in `escaped_byte_buf`
  (#610).

Both commits' regression tests (`tests/607_quadratic_float_parsing.rs`,
`tests/609_quadratic_escaped_string_parsing.rs`) are registered in this copy's
`Cargo.toml`, whose published form lists test targets explicitly.

Delete this copy, its `[patch]` entry and its `exclude` entry once a ron
release contains both commits.
