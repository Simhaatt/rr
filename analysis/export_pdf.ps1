# Opens the report built by 07_report.py in Microsoft Word, fills in the contents page numbers, and exports the PDF.
# Usage (from the repository root):  powershell -ExecutionPolicy Bypass -File analysis\export_pdf.ps1
$root = Split-Path -Parent $PSScriptRoot
$docx = Join-Path $root 'submission\SupeRR_Selector_R1_Methodology_Report.docx'
[string]$pdf = Join-Path $root 'submission\SupeRR_Selector_R1_Methodology_Report.pdf'
$w = New-Object -ComObject Word.Application; $w.Visible = $false
try {
  $d = $w.Documents.Open($docx)
  foreach ($i in -20, -21) {            # contents styles TOC 1 and TOC 2: compact, so the contents fit on one page
    $s = $d.Styles.Item($i); $s.ParagraphFormat.SpaceBefore = 0; $s.ParagraphFormat.SpaceAfter = 0
    $s.ParagraphFormat.LineSpacingRule = 0; $s.Font.Size = 9
  }
  $d.Styles.Item(-20).Font.Bold = $true
  $d.TablesOfContents.Item(1).Update(); $d.Repaginate(); $d.TablesOfContents.Item(1).UpdatePageNumbers()
  $d.Save()
  $d.SaveAs([ref]$pdf, [ref]17)
  "pages $($d.ComputeStatistics(2))"
  $d.Close()
} finally { $w.Quit() }
