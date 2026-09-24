"""Fail CI on broken portal references, duplicate IDs and unlabeled controls."""
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
import re
import sys


class Portal(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []
        self.labels = set()
        self.controls = []
        self.anchors = []
        self.assets = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            self.ids.append(attrs['id'])
        if tag == 'label' and 'for' in attrs:
            self.labels.add(attrs['for'])
        if tag in {'input', 'select', 'textarea'}:
            self.controls.append(attrs)
        if tag == 'a':
            self.anchors.append(attrs.get('href', ''))
        if tag in {'script', 'link', 'img'}:
            self.assets.append(attrs.get('src', attrs.get('href', '')))


def audit(path):
    html = path.read_text(encoding='utf-8')
    portal = Portal()
    portal.feed(html)
    errors = []
    ids = set(portal.ids)
    errors.extend(f'duplicate ID: {key}' for key, n in Counter(portal.ids).items() if n > 1)
    referenced = set(re.findall(r'getElementById\(["\']([^"\']+)["\']\)', html))
    errors.extend(f'missing script target: {key}' for key in sorted(referenced - ids))
    for attrs in portal.controls:
        if not (attrs.get('id') in portal.labels or attrs.get('aria-label') or attrs.get('aria-labelledby')):
            errors.append(f'unlabeled control: {attrs.get("id", "(no id)")}')
    for target in portal.anchors + portal.assets:
        if not target or '://' in target or target.startswith(('mailto:', 'data:')):
            continue
        if target.startswith('#'):
            if target[1:] not in ids:
                errors.append(f'missing anchor: {target}')
        else:
            local = path.parent / target.split('#')[0].split('?')[0]
            # GitHub Pages converts Markdown documents to .html.
            if not local.exists() and not (local.suffix == '.html' and local.with_suffix('.md').exists()):
                errors.append(f'missing local resource: {target}')
    scripts = '\n'.join(re.findall(r'<script(?:\s[^>]*)?>(.*?)</script>', html, re.S))
    for handler in re.findall(r'\bon[a-z]+=["\']([^"\']+)["\']', html):
        name = handler.split('(')[0].strip()
        if not re.search(r'\bfunction\s+' + re.escape(name) + r'\s*\(', scripts):
            errors.append(f'undefined handler: {name}')
    return errors


if __name__ == '__main__':
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / 'docs/index.html'
    errors = audit(path)
    for error in errors:
        print(error)
    print(f'Portal audit: {len(errors)} errors')
    raise SystemExit(bool(errors))
