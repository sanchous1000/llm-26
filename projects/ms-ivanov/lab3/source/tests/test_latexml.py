import shutil

import pytest

from source.parsing.latexml import LatexmlConverter

pytestmark = pytest.mark.skipif(
    shutil.which("pandoc") is None, reason="pandoc not installed"
)

BASE = "https://ar5iv.labs.arxiv.org/html/2001.00001"


def math(tex: str, display: str = "inline") -> str:
    return (
        f'<math alttext="{tex}" class="ltx_Math" display="{display}">'
        f'<semantics><mi>x</mi><annotation encoding="application/x-tex">{tex}</annotation>'
        "</semantics></math>"
    )


def paper(body: str) -> str:
    padding = '<p class="ltx_p">filler text</p>' * 12
    return (
        '<html><body><article class="ltx_document">'
        f"{body}{padding}</article></body></html>"
    )


def convert(body: str) -> str:
    return LatexmlConverter().convert_html(paper(body), BASE).markdown


def test_title_abstract_and_numbered_sections():
    markdown = convert(
        '<h1 class="ltx_title ltx_title_document">My <em>Title</em>'
        '<span class="ltx_pubnotes">Thanks: funding</span></h1>'
        '<div class="ltx_authors">Alice, Bob</div>'
        '<div class="ltx_abstract"><h6 class="ltx_title ltx_title_abstract">Abstract</h6>'
        '<p class="ltx_p">We study things.</p></div>'
        '<section class="ltx_section"><h2 class="ltx_title ltx_title_section">'
        '<span class="ltx_tag ltx_tag_section">1 </span>Intro</h2>'
        '<div class="ltx_para"><p class="ltx_p">Body.</p></div>'
        '<section class="ltx_subsection"><h3 class="ltx_title ltx_title_subsection">'
        '<span class="ltx_tag ltx_tag_subsection">1.1 </span>Detail</h3></section></section>'
    )

    assert markdown.startswith("# My Title\n")
    assert "Thanks" not in markdown and "Alice" not in markdown
    assert "## Abstract\n\nWe study things." in markdown
    assert "## 1 Intro" in markdown and "### 1.1 Detail" in markdown


def test_inline_math_keeps_tex_unescaped():
    markdown = convert(f'<p class="ltx_p">Let {math("x_1^{*} * y")} be small.</p>')

    assert "Let $x_1^{*} * y$ be small." in markdown


def test_equation_table_becomes_display_math_with_tag():
    table = (
        '<table class="ltx_equation ltx_eqn_table"><tbody>'
        '<tr class="ltx_equation ltx_eqn_row"><td class="ltx_eqn_cell"></td>'
        f'<td class="ltx_eqn_cell">{math("E=mc^2", "block")}</td>'
        '<td class="ltx_eqn_cell ltx_eqn_eqno"><span class="ltx_tag">(3)</span></td>'
        "</tr></tbody></table>"
    )

    markdown = convert(table)

    assert "$$\nE=mc^2 \\tag{3}\n$$" in markdown


def test_math_in_table_cell_escapes_pipes():
    table = (
        '<table class="ltx_tabular"><tr><td>A</td><td>B</td></tr>'
        f"<tr><td>{math('|x|')}</td><td>2</td></tr></table>"
    )

    markdown = convert(table)

    assert "$\\vert x\\vert $" in markdown
    assert "| A" in markdown


def test_theorem_and_proof_get_bold_lead_in_without_italics():
    markdown = convert(
        '<div class="ltx_theorem ltx_theorem_lemma"><h6 class="ltx_title ltx_runin '
        'ltx_title_theorem"><span class="ltx_tag ltx_tag_theorem">Lemma 1</span>'
        '<span class="ltx_text ltx_font_bold">.</span></h6>'
        '<div class="ltx_para"><p class="ltx_p"><span class="ltx_text ltx_font_italic">'
        "All maps are nice.</span></p></div></div>"
        '<div class="ltx_proof"><h6 class="ltx_title ltx_runin ltx_title_proof">Proof.</h6>'
        '<div class="ltx_para"><p class="ltx_p">Trivial.</p></div></div>'
    )

    assert "**Lemma 1.** All maps are nice." in markdown
    assert "**Proof.** Trivial." in markdown
    assert "*All maps" not in markdown


def test_citations_and_cross_references_become_plain_text():
    markdown = convert(
        '<p class="ltx_p">See <cite class="ltx_cite">(<a class="ltx_ref" href="#bib.bib1">'
        'Doe 2020</a>)</cite> and <a class="ltx_ref" href="#S2">Section 2</a>, '
        '<a href="/abs/1234.5678">arXiv</a>.</p>'
    )

    assert (
        "See (Doe 2020) and Section 2, [arXiv](https://ar5iv.labs.arxiv.org/abs/1234.5678)."
        in markdown
    )


def test_footnote_is_inlined_where_referenced():
    markdown = convert(
        '<p class="ltx_p">Claim<span class="ltx_note ltx_role_footnote"><sup class="ltx_note_mark">1</sup>'
        '<span class="ltx_note_outer"><span class="ltx_note_content"><sup class="ltx_note_mark">1</sup>'
        '<span class="ltx_tag ltx_tag_note">1</span>Extra detail.</span></span></span> holds.</p>'
    )

    assert "Claim [Footnote: Extra detail.] holds." in markdown


def test_figure_keeps_caption_text_and_absolute_image_links():
    markdown = convert(
        '<figure class="ltx_figure"><img alt="Refer to caption" src="/html/2001.00001/assets/a.png" '
        'width="10" height="10"><img src="data:image/png;base64,AAAA">'
        '<figcaption class="ltx_caption"><span class="ltx_tag ltx_tag_figure">Figure 1: </span>'
        "Our pipeline.</figcaption></figure>"
    )

    assert (
        "![Figure 1](https://ar5iv.labs.arxiv.org/html/2001.00001/assets/a.png)"
        in markdown
    )
    assert "**Figure 1:** Our pipeline." in markdown
    assert "base64" not in markdown


def test_table_with_colspan_and_rowspan_is_rectangular():
    table = (
        '<table class="ltx_tabular"><tr><td colspan="2">Head</td><td>C</td></tr>'
        '<tr><td rowspan="2">a</td><td>b</td><td>c</td></tr>'
        "<tr><td>d</td><td>e</td></tr></table>"
    )

    lines = [line for line in convert(table).splitlines() if line.startswith("|")]

    cells = [[c.strip() for c in line.strip("|").split("|")] for line in lines]
    assert cells[0] == ["Head", "", "C"]
    assert cells[2] == ["a", "b", "c"]
    assert cells[3] == ["", "d", "e"]


def test_lists_keep_structure_and_drop_bullet_tags():
    markdown = convert(
        '<ul class="ltx_itemize"><li class="ltx_item"><span class="ltx_tag ltx_tag_item">•</span>'
        '<div class="ltx_para"><p class="ltx_p">first</p></div></li>'
        '<li class="ltx_item"><span class="ltx_tag ltx_tag_item">•</span>'
        '<div class="ltx_para"><p class="ltx_p">second</p></div></li></ul>'
        '<ol class="ltx_enumerate"><li class="ltx_item"><span class="ltx_tag ltx_tag_item">(a)</span>'
        '<div class="ltx_para"><p class="ltx_p">alpha</p></div></li></ol>'
    )

    assert "- first\n- second" in markdown
    assert "- (a) alpha" in markdown


def test_algorithm_listing_becomes_fenced_code_with_inline_math():
    markdown = convert(
        '<div class="ltx_listing"><div class="ltx_listingline">for i in range: '
        f'{math("x_i")} += 1</div><div class="ltx_listingline">return</div></div>'
    )

    assert "```\nfor i in range: $x_i$ += 1\nreturn\n```" in markdown


def test_bibliography_lists_entries_with_labels():
    markdown = convert(
        '<section class="ltx_bibliography"><h2 class="ltx_title ltx_title_bibliography">References</h2>'
        '<ul class="ltx_biblist"><li class="ltx_bibitem"><span class="ltx_tag ltx_tag_bibitem">[1]</span>'
        '<span class="ltx_bibblock">A. Author. </span><span class="ltx_bibblock">A paper. '
        "<em>Venue</em>, 2020. </span></li></ul></section>"
    )

    assert "## References" in markdown
    assert "- **[1]** A. Author. A paper. *Venue*, 2020." in markdown


def test_references_can_be_omitted():
    html = paper(
        '<section class="ltx_bibliography"><h2 class="ltx_title">References</h2>'
        '<ul class="ltx_biblist"><li class="ltx_bibitem">x</li></ul></section>'
    )

    markdown = (
        LatexmlConverter(include_references=False).convert_html(html, BASE).markdown
    )

    assert "References" not in markdown


def test_converter_name_reflects_options():
    assert LatexmlConverter().name != LatexmlConverter(include_references=False).name


def test_non_paper_html_is_rejected():
    from source.parsing.errors import ConversionError

    with pytest.raises(ConversionError):
        LatexmlConverter().convert_html("<html><body>nothing</body></html>", BASE)


def test_nested_font_reset_splits_bold_run():
    markdown = convert(
        '<p class="ltx_p"><span class="ltx_font_bold">Left:<span class="ltx_font_medium"> normal '
        "text </span>Right:</span> end.</p>"
    )

    assert "**Left:** normal text **Right:** end." in markdown


def test_each_standalone_image_gets_its_own_paragraph():
    markdown = convert(
        '<div><img src="/a.png" alt="x"><img src="/b.png" alt="y"></div>'
    )

    assert (
        "![x](https://ar5iv.labs.arxiv.org/a.png)\n\n![y](https://ar5iv.labs.arxiv.org/b.png)"
        in markdown
    )
