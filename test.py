import torch
import Levenshtein
from difflib import SequenceMatcher
from ctcdecode import CTCBeamDecoder
import numpy as np

def char_err_rate(s1, s2):
    s1 = s1.replace(' ', '')
    s2 = s2.replace(' ', '')

    dist = Levenshtein.distance(s1, s2)

    return dist / len(s2)

def test(model, test_loader, ocr_dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ratios = []
    lv_ratios = []

    BLANK = ocr_dataset.get_num_classes()-1

    with torch.no_grad():
        for ((x, input_lengths),(y,target_lengths)) in test_loader:
            print("Run eval")
            x = x.to(device)
            
            outputs = model.forward(x)
            outputs = outputs.permute(1, 0, 2)
            
            decoder = CTCBeamDecoder(ocr_dataset.char_vec,
                                    blank_id=BLANK,
                                    log_probs_input=True)

            output, scores, ts, out_seq_len = decoder.decode(outputs.data, 
                                                    torch.IntTensor(input_lengths))

            results = []
        
            for b, batch in enumerate(output):
                size = out_seq_len[b][0]
                dec = batch[0]

                text = ''
                if size > 0:
                    text = ocr_dataset.get_decoded_label(dec[0:size])
                
                results.append(text)
            
            
            ptr = 0
            for i, p in enumerate(target_lengths):
                yi = y[ptr:ptr+p]
                
                s1 = results[i]
                s2 = ocr_dataset.get_decoded_label(yi)

                ratios.append(SequenceMatcher(None, s1, s2).quick_ratio())
                
                lv_ratios.append(char_err_rate(s1, s2))

                ptr += p   

    print("SequenceMatcher acc:", np.mean(ratios), np.std(ratios))
    print("Levenshtein acc:", np.mean(lv_ratios), np.std(lv_ratios))