from __future__ import annotations
import logging
import re
import warnings
from bs4 import BeautifulSoup, Tag, MarkupResemblesLocatorWarning
from dataclasses import dataclass

log = logging.getLogger("javadoc-mcp.parser")
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)


@dataclass
class Symbol:
    fqn: str
    kind: str
    name: str
    package: str
    signature: str | None = None
    summary: str | None = None
    description: str | None = None
    html_path: str | None = None
    source_jar: str | None = None


def _strip_html(html: str) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(["br", "li", "dt", "dd"]):
        tag.insert_before("\n")
    for tag in soup.find_all(True):
        tag.replace_with(tag.get_text())
    text = soup.get_text()
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def _extract_block_text(parent: Tag | None, tag_name: str = "div", klass: str = "block") -> str:
    if not parent:
        return ""
    block = parent.find(tag_name, class_=klass)
    if not block:
        return ""
    return _strip_html(block.decode_contents())


def _get_anchor_section(body: Tag, anchor_id: str) -> Tag | None:
    el = body.find("section", id=anchor_id)
    if el:
        return el
    el = body.find("div", id=anchor_id)
    return el


def parse_class_page(html: str, html_path: str, jar_path: str) -> list[Symbol]:
    soup = BeautifulSoup(html, "html.parser")
    body = soup.body
    if not body:
        log.debug(f"[parser] No body found in {html_path}")
        return []

    page_class = body.get("class", [])
    is_class = "class-declaration-page" in page_class
    is_interface = "interface-declaration-page" in page_class
    is_enum = "enum-declaration-page" in page_class

    if not (is_class or is_interface or is_enum):
        log.debug(f"[parser] Not a class/interface/enum page: {html_path} (classes: {page_class})")
        return []

    kind = "enum" if is_enum else "interface" if is_interface else "class"

    class_name = ""
    package = ""

    cap_el = body.find("h1")
    if cap_el:
        raw = cap_el.get_text(strip=True)
        for prefix in ("Class ", "Interface ", "Enum ", "Annotation Type ", "Record "):
            if raw.startswith(prefix):
                raw = raw[len(prefix):]
        class_name = raw

    if not class_name:
        log.debug(f"[parser] No class name found in {html_path}")
        return []

    if "<" in class_name:
        class_name = class_name.split("<")[0].strip()

    # Javadoc 21: class-description is a <section>, not a <div>
    desc_section = body.find("section", class_="class-description")
    if not desc_section:
        desc_section = body.find("div", class_="description")

    class_summary = _extract_block_text(desc_section)
    class_description = _extract_block_text(desc_section)

    type_sig_el = body.find("div", class_="type-signature")
    type_signature = type_sig_el.get_text(strip=True) if type_sig_el else None

    pkg_el = body.find("section", class_="package")
    if pkg_el:
        pkg_text = pkg_el.get_text(strip=True)
        for prefix in ("Package ", "Package"):
            if pkg_text.startswith(prefix):
                pkg_text = pkg_text[len(prefix):].strip()
        if pkg_text:
            package = pkg_text
    if not package:
        path_parts = html_path.replace(".html", "").split("/")
        if len(path_parts) > 1:
            package = ".".join(path_parts[:-1])

    fqn = f"{package}.{class_name}" if package else class_name

    symbols: list[Symbol] = []

    class_sym = Symbol(
        fqn=fqn,
        kind=kind,
        name=class_name,
        package=package,
        signature=type_signature,
        summary=class_summary[:500] if class_summary else None,
        description=class_description,
        html_path=html_path,
        source_jar=jar_path,
    )
    symbols.append(class_sym)

    fields = _parse_member_section(body, "field-summary", "field-detail", fqn, package, jar_path, html_path)
    constructors = _parse_member_section(body, "constructor-summary", "constructor-detail", fqn, package, jar_path, html_path)
    methods = _parse_member_section(body, "method-summary", "method-detail", fqn, package, jar_path, html_path)

    symbols.extend(fields)
    symbols.extend(constructors)
    symbols.extend(methods)

    return symbols


def _get_detail_text(detail_section: Tag | None, method_name: str, kind: str) -> str:
    if not detail_section:
        return ""
    if kind == "field":
        sec = detail_section.find("section", id=method_name)
        if sec:
            block = sec.find("div", class_="block")
            if block:
                return _strip_html(block.decode_contents())
        return ""
    prefix = f"{method_name}("
    for sec in detail_section.find_all("section", class_="detail"):
        sec_id = sec.get("id", "")
        if sec_id.startswith(prefix):
            block = sec.find("div", class_="block")
            if block:
                return _strip_html(block.decode_contents())
    return ""


def _parse_member_section(
    body: Tag,
    summary_id: str,
    detail_id: str,
    parent_fqn: str,
    package: str,
    jar_path: str,
    html_path: str,
) -> list[Symbol]:
    symbols: list[Symbol] = []

    member_kinds = {
        "field-summary": "field",
        "constructor-summary": "constructor",
        "method-summary": "method",
    }
    kind = member_kinds.get(summary_id, "method")

    summary_section = body.find("section", id=summary_id)
    if not summary_section:
        return symbols

    detail_section = body.find("section", id=detail_id)

    summary_table = summary_section.find("div", class_="summary-table")
    if not summary_table:
        return symbols

    for row in summary_table.find_all("div", class_="col-second"):
        name_link = row.find("a", class_="member-name-link")
        if not name_link:
            continue
        member_name = name_link.get_text(strip=True)
        if not member_name:
            continue

        clean_name = member_name.split("(")[0].strip()

        # Get signature from the col-second content
        sig = row.get_text(strip=True)

        # Get description from the adjacent col-last
        col_last = row.find_next_sibling("div", class_="col-last")
        member_summary = ""
        if col_last:
            block = col_last.find("div", class_="block")
            if block:
                member_summary = _strip_html(block.decode_contents())

        # Try to get full detail
        full_desc = _get_detail_text(detail_section, clean_name, kind)
        if not full_desc:
            full_desc = member_summary

        if kind == "method":
            fqn_str = f"{parent_fqn}#{clean_name}"
        elif kind == "constructor":
            fqn_str = f"{parent_fqn}#<init>"
            clean_name = parent_fqn.split(".")[-1]
        else:
            fqn_str = f"{parent_fqn}#{clean_name}"

        sym = Symbol(
            fqn=fqn_str,
            kind=kind,
            name=clean_name,
            package=package,
            signature=sig,
            summary=member_summary[:500] if member_summary else None,
            description=full_desc,
            html_path=html_path,
            source_jar=jar_path,
        )
        symbols.append(sym)

    return symbols
