# 下载并解压 RDK X5 OE SDK 交付包（v1.2.8，含 model_zoo PTQ 示例）
# 用法: 在宿主机 PowerShell 中执行  .\scripts\download_oe_sdk.ps1
# 支持断点续传: 下载中断后重新执行即可继续

$ErrorActionPreference = "Stop"

$oeSdkDir = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\oe_sdk"))
New-Item -ItemType Directory -Force -Path $oeSdkDir | Out-Null

# 官方文档站地址（2026-09-03 更新），404 则回退旧文件名
$candidates = @(
    "https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/oe_x5/1.2.8/D-Robotics_x5_open_explorer_v1.2.8-py310_20240926.tar.gz",
    "https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/oe_x5/1.2.8/horizon_x5_open_explorer_v1.2.8-py310_20240926.tar.gz"
)

$url = $null
foreach ($u in $candidates) {
    Write-Host "探测: $u"
    $code = & curl.exe -s -o NUL -w "%{http_code}" --head -L $u
    if ($code -eq "200") { $url = $u; break }
    Write-Host "  -> HTTP $code，尝试下一个地址"
}
if (-not $url) { throw "两个官方地址均不可用，请检查网络或到官方文档确认最新地址" }

$fileName = [System.IO.Path]::GetFileName($url)
$tarPath = Join-Path $oeSdkDir $fileName

Write-Host "下载: $url"
Write-Host "保存: $tarPath （支持断点续传，中断后重新运行即可）"
& curl.exe -L -C - --retry 5 --retry-delay 3 -o $tarPath $url
if ($LASTEXITCODE -ne 0) { throw "下载失败（curl exit=$LASTEXITCODE），重新运行本脚本可续传" }

Write-Host "解压到: $oeSdkDir （注：该包可能为纯 POSIX tar，tar -xf 通吃两种格式）"
& tar -xf $tarPath -C $oeSdkDir
if ($LASTEXITCODE -ne 0) {
    Write-Warning "tar 返回非零（Windows 无法创建包内 6 个符号链接所致，属预期现象，真实内容不受影响；容器启动自检会自动补建这些链接）"
}

Write-Host ""
Write-Host "=== 完成。oe_sdk/ 目录内容 ==="
Get-ChildItem $oeSdkDir | ForEach-Object { Write-Host $_.Name }
Write-Host ""
Write-Host "=== SDK 包顶层内容 ==="
Get-ChildItem $oeSdkDir -Directory | ForEach-Object {
    Write-Host "[$($_.Name)]"
    Get-ChildItem $_.FullName | Select-Object -First 15 | ForEach-Object { Write-Host "  $($_.Name)" }
}
