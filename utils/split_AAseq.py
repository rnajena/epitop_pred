# with open("/home/le86qiz/Documents/Konrad/prediction_pipeline/raptorx_pipeline/NP_220200.1.faa","r") as input_file:
with open("/home/le86qiz/Documents/Konrad/stiko/second_prediction/bacteria/bacteria_gram-/results/single_fastas/sp_P14283_PERT_BORPE.faa","r") as input_file:
    seq = ""
    for line in input_file:
        if line.startswith(">"):
            continue
        else:
            if line.endswith("\n"):
                line = line[:-1]
            seq += line
    with open("/home/go96bix/projects/epitop_pred/220200_test.csv", "w") as output_file:
        for i in range(0,len(seq)-50):
            output_file.write(f"{i+25}\t{seq[i:i+50]}\n")

