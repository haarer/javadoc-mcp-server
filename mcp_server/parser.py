from __future__ import annotations
import re
from bs4 import BeautifulSoup, Tag
from dataclasses import dataclass


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
        tag.decompose()
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


def parse_class_page(html: str, html_path: str, jar_path: str) -> list[Symbol]:
    soup = BeautifulSoup(html, "html.parser")
    body = soup.body
    if not body:
        return []

    page_class = body.get("class", [])
    is_class = "class-declaration-page" in page_class
    is_interface = "interface-declaration-page" in page_class
    is_enum = "enum-declaration-page" in page_class

    if not (is_class or is_interface or is_enum):
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
        return []

    # Strip generic type parameters from class name
    if "<" in class_name:
        class_name = class_name.split("<")[0].strip()

    desc_section = body.find("div", class_="description")
    class_summary = _extract_block_text(desc_section)
    class_description = _extract_block_text(desc_section)

    type_sig_el = body.find("div", class_="type-signature")
    type_signature = type_sig_el.get_text(strip=True) if type_sig_el else None

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
    summary_table = body.find("section", id=summary_id)
    detail_section = body.find("section", id=detail_id)

    if not summary_table:
        return symbols

    summary_ul = summary_table.find("ul")
    if not summary_ul:
        return symbols

    member_kinds = {
        "field-summary": "field",
        "constructor-summary": "constructor",
        "method-summary": "method",
    }
    kind = member_kinds.get(summary_id, "method")

    for li in summary_ul.find_all("li"):
        name_el = li.find("span", class_="member-name")
        if not name_el:
            continue

        member_name = ""
        sig = ""

        name_span = name_el.find("span", class_="member-name-link")
        if name_span:
            member_name = name_span.get_text(strip=True)

        desc_div = name_el.find_next_sibling("div", class_="block")
        member_summary = ""
        if desc_div:
            member_summary = _strip_html(desc_div.decode_contents())

        if detail_section and member_name:
            anchor_id = member_name.split("(")[0].replace("$", "_")
            detail_li = detail_section.find("li", id=anchor_id)
            if not detail_li:
                detail_li = detail_section.find("li", id=f"_{anchor_id}")
            if detail_li:
                detail_block = detail_li.find("div", class_="block")
                if detail_block:
                    full_desc = _strip_html(detail_block.decode_contents())
                    if full_desc and len(full_desc) > len(member_summary):
                        member_summary = full_desc

        if not member_name:
            continue

        clean_name = member_name.split("(")[0].strip()
        sig = name_el.get_text(strip=True)

        if kind == "method":
            fqn = f"{parent_fqn}#{clean_name}"
        elif kind == "constructor":
            fqn = f"{parent_fqn}#<init>"
            clean_name = parent_fqn.split(".")[-1]
        else:
            fqn = f"{parent_fqn}#{clean_name}"

        sym = Symbol(
            fqn=fqn,
            kind=kind,
            name=clean_name,
            package=package,
            signature=sig,
            summary=member_summary[:500] if member_summary else None,
            description=member_summary,
            html_path=html_path,
            source_jar=jar_path,
        )
        symbols.append(sym)

    return symbols
