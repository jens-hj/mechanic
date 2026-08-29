[CmdletBinding()]
param(
    [ValidateSet("build", "run", "test", "check")]
    [string]$Command = "run",

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$AppArguments
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($env:OS -ne "Windows_NT") {
    throw "scripts/dev.ps1 is for native Windows. Use 'nix develop' on macOS or Linux."
}

foreach ($RequiredCommand in @("git", "rustup", "cargo")) {
    if (-not (Get-Command $RequiredCommand -ErrorAction SilentlyContinue)) {
        throw @"
'$RequiredCommand' is required. Install Git, rustup from https://rustup.rs,
and Visual Studio 2022 Build Tools with the 'Desktop development with C++'
workload, then open a new PowerShell window.
"@
    }
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Executable,

        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

$RepositoryRoot = Split-Path -Parent $PSScriptRoot
Push-Location $RepositoryRoot
try {
    Invoke-Checked "rustup" @("show")

    switch ($Command) {
        "build" {
            Invoke-Checked "cargo" @("build", "--workspace")
        }
        "run" {
            $CargoArguments = @("run", "-p", "mechanic-app", "--") + $AppArguments
            Invoke-Checked "cargo" $CargoArguments
        }
        "test" {
            Invoke-Checked "cargo" @("test", "--workspace")
        }
        "check" {
            Invoke-Checked "cargo" @("fmt", "--all", "--", "--check")
            Invoke-Checked "cargo" @("check", "--workspace", "--all-targets")
            Invoke-Checked "cargo" @("clippy", "--workspace", "--all-targets", "--", "-D", "warnings")
        }
    }
}
finally {
    Pop-Location
}
