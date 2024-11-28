out_file=$1
snp_file=$2
snp_list=$(cat $snp_file | tr "\n" "," | sed "s/,$//" | sed "s/,/\\\n/g")
# echo $out_file
curl \
    -k \
    -H "Content-Type: application/json" \
    -X POST \
    -o $out_file \
    -d '{"snps":"'$snp_list'", "pop": "GBR", "r2_d": "r2", "genome_build": "grch37"}' \
    'https://ldlink.nih.gov/LDlinkRest/ldmatrix?token=81b0b1b59294'
