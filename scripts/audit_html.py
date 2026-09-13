import re

with open("docs/index.html", "r", encoding="utf-8") as f:
    html = f.read()

# Extract script blocks
scripts = re.findall(r"<script(?:\s[^>]*)?>(.*?)</script>", html, re.DOTALL)
print(f"Total script tags: {len(scripts)}")

# All getElementById
js_ids = set(re.findall(r'getElementById\(["\']([^"\']+)["\']\)', html))
# All id= in HTML tags
html_ids = set(re.findall(r'\bid=["\']([^"\']+)["\']', html))

missing = js_ids - html_ids
print("JS IDs referenced:", len(js_ids))
print("HTML IDs defined:", len(html_ids))
print("Missing IDs:", missing)

# Check inline event handlers (e.g. onchange="foo()", onclick="bar()")
handlers = re.findall(r'\bon[a-z]+=["\']([^"\']+)["\']', html)
print(f"Total inline event handlers: {len(handlers)}")
for h in handlers:
    func_name = h.split("(")[0].strip()
    # Check if func_name is defined in script
    found = False
    for s in scripts:
        if f"function {func_name}" in s or f"{func_name} =" in s or f"{func_name}=" in s:
            found = True
            break
    if not found:
        print(f"WARNING: Handler '{h}' calls function '{func_name}' which may NOT be defined!")
