while ($true){
    try{
        python .\button.py
    } catch{
        $env:BLINKA_MCP2221=1
    }
}
