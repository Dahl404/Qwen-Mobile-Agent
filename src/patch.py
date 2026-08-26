#!/usr/bin/env python3
"""
patch_attn_buffer.py - Revert attn_worker scores buffer to safe size (n_ctx + 512).
Run: python3 patch_attn_buffer.py nn.c
"""

import sys
import re

def patch_attn_worker(content):
    # Find the attn_worker function body
    func_pattern = r'(static void attn_worker\(void \*arg, int i0, int i1\)\s*\{.*?)\n\}'
    match = re.search(func_pattern, content, re.DOTALL)
    if not match:
        print("attn_worker function not found!", file=sys.stderr)
        return content

    func_body = match.group(1)

    # Replace the need_scores block with a simple fixed size
    new_calc = """
    size_t need_scores = (size_t)(c->n_ctx + 512);
    """
    # The original block we want to replace:
    #     size_t need_scores = (size_t)c->n_ctx + 512;
    #     if ((size_t)c->n_live + 512 > need_scores)
    #         need_scores = (size_t)c->n_live + 512;
    # We'll replace it with the single line above.
    pattern = r'(size_t need_scores = .*?;\s*if \(\(size_t\)c->n_live \+ 512 > need_scores\)\s*need_scores = \(size_t\)c->n_live \+ 512;)'
    replacement = new_calc.strip()
    new_body = re.sub(pattern, replacement, func_body, flags=re.DOTALL)
    if new_body == func_body:
        print("Could not find need_scores block; maybe already patched?", file=sys.stderr)
        # Fallback: try to find any definition of need_scores and replace the whole assignment
        pattern2 = r'size_t need_scores = .*?;'
        replacement2 = '    size_t need_scores = (size_t)(c->n_ctx + 512);'
        new_body = re.sub(pattern2, replacement2, func_body, count=1)
        if new_body == func_body:
            print("No suitable pattern found, please patch manually.", file=sys.stderr)
            return content

    # Replace the whole function body
    new_content = content[:match.start(1)] + new_body + content[match.end(1):]
    return new_content

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 patch_attn_buffer.py nn.c", file=sys.stderr)
        sys.exit(1)
    fname = sys.argv[1]
    with open(fname, 'r') as f:
        content = f.read()
    patched = patch_attn_worker(content)
    if patched != content:
        with open(fname, 'w') as f:
            f.write(patched)
        print(f"Patched {fname} successfully.")
    else:
        print("No changes made.")

if __name__ == "__main__":
    main()
