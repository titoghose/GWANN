import requests 

def STRING_PPI_enrichment(gene_list:list, analysis_name:str='') -> dict:

    string_api_url = "https://version-11-5.string-db.org/api"
    output_format = "tsv"
    method = "ppi_enrichment"

    request_url = "/".join([string_api_url, output_format, method])

    params = {
        "identifiers" : "%0d".join(gene_list),
        "species" : 9606, # NCBI species ID for human
        "caller_identity" : analysis_name}
    
    response = requests.post(request_url, data=params)

    lines = response.text.strip().split("\n")
    header = lines[0]
    out_dict = {h:None for h in header.split("\t")}
    for line in lines[1:]:
        for hi, h in enumerate(out_dict.keys()):
            out_dict[h] = line.split("\t")[hi]
    
    return out_dict

if __name__ == '__main__':
    glist = []
    STRING_PPI_enrichment(glist, analysis_name='test')