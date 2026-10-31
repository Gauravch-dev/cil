param([string]$DocxPath, [string]$PdfPath)
$ErrorActionPreference = 'Stop'
$taskDocument = $null
$taskWord = $null
$taskWordInitialDocs = -1
$taskWordOldAlerts = $null
try {
    $taskInput = (Resolve-Path -LiteralPath $DocxPath).Path
    $taskOutput = [System.IO.Path]::GetFullPath($PdfPath)
    $taskWord = New-Object -ComObject Word.Application
    $taskWordInitialDocs = $taskWord.Documents.Count
    $taskWordOldAlerts = $taskWord.DisplayAlerts
    if ($taskWordInitialDocs -eq 0) { $taskWord.Visible = $false }
    $taskWord.DisplayAlerts = 0
    $taskDocument = $taskWord.Documents.Open($taskInput, $false, $true)
    $taskDocument.Repaginate()
    $taskPageCount = $taskDocument.ComputeStatistics(2)
    $taskDocument.ExportAsFixedFormat($taskOutput, 17)
    Write-Output ('Rendered using native Word: ' + $taskOutput)
    Write-Output ('Pages: ' + $taskPageCount)
} finally {
    if ($null -ne $taskDocument) {
        $taskDocument.Close(0)
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($taskDocument)
    }
    if ($null -ne $taskWord) {
        if ($null -ne $taskWordOldAlerts) { $taskWord.DisplayAlerts = $taskWordOldAlerts }
        if ($taskWordInitialDocs -eq 0) { $taskWord.Quit(0) }
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($taskWord)
    }
}
