while ($true){
    python .\button.py Fail 2> $null
    # if ($LASTEXITCODE -ne 0){
    #     $env:BLINKA_MCP2221=1
    # }
}
