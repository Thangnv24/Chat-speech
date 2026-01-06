Get-Content .env | ForEach-Object {
    $line = $_.Trim()
    
    if (-not $line) { return }

    if ($line.StartsWith("#")) { return }
    
    if ($line -like '*=*') {
        $name, $value = $line.Split("=", 2)
        
        if (-not $name.Trim()) { return }
        $cleanValue = $value.Trim().TrimStart("'").TrimEnd("'").TrimStart('"').TrimEnd('"')
        Set-Item -Path Env:$($name.Trim()) -Value $cleanValue
        
        Write-Host "Set Environment Variable: $($name.Trim())"
    }
}