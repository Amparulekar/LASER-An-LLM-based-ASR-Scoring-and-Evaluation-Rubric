import difflib
import csv

def align_hindi_sentences(s1: str, s2: str):
    # 1. Tokenize on whitespace
    tokens1 = s1.split()
    tokens2 = s2.split()

    # 2. SequenceMatcher over token lists
    matcher = difflib.SequenceMatcher(None, tokens1, tokens2)
    raw_pairs = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            # exact matches → one-to-one
            for idx1, idx2 in zip(range(i1, i2), range(j1, j2)):
                raw_pairs.append((tokens1[idx1], tokens2[idx2]))
        else:
            span1 = tokens1[i1:i2]
            span2 = tokens2[j1:j2]
            n, m = len(span1), len(span2)

            # --- 1) 1→1 mappings first ---
            core = min(n, m)
            for idx in range(core):
                raw_pairs.append((span1[idx], span2[idx]))

            left1 = span1[core:]
            left2 = span2[core:]

            # --- 2) compound 2→1 or 1→2 next ---
            # only if both sides still have tokens
            if left1 and left2:
                if len(left1) > len(left2):
                    diff = len(left1) - len(left2)
                    for _ in range(diff):
                        raw_pairs.append((' '.join(left1[:2]), left2[0]))
                        left1 = left1[2:]
                        left2 = left2[1:]
                elif len(left2) > len(left1):
                    diff = len(left2) - len(left1)
                    for _ in range(diff):
                        raw_pairs.append((left1[0], ' '.join(left2[:2])))
                        left1 = left1[1:]
                        left2 = left2[2:]

            # --- 3) any remaining 1→1 after compounds ---
            core2 = min(len(left1), len(left2))
            for idx in range(core2):
                raw_pairs.append((left1[idx], left2[idx]))

            # --- 4) null mappings for pure leftovers ---
            for w1 in left1[core2:]:
                raw_pairs.append((w1, 'null'))
            for w2 in left2[core2:]:
                raw_pairs.append(('null', w2))

    # --- 5) Merge null‐pairs into neighbors if possible ---
    merged = []
    i = 0
    while i < len(raw_pairs):
        l, r = raw_pairs[i]
        if r == 'null' and l != 'null':
            # deletion: try merge left word into previous or next
            if merged:
                prev_l, prev_r = merged[-1]
                merged[-1] = (f"{prev_l} {l}", prev_r)
            elif i + 1 < len(raw_pairs):
                nxt_l, nxt_r = raw_pairs[i+1]
                merged.append((f"{l} {nxt_l}", nxt_r))
                i += 1
            else:
                merged.append((l, 'null'))
        elif l == 'null' and r != 'null':
            # insertion: try merge right word into previous or next
            if merged:
                prev_l, prev_r = merged[-1]
                merged[-1] = (prev_l, f"{prev_r} {r}")
            elif i + 1 < len(raw_pairs):
                nxt_l, nxt_r = raw_pairs[i+1]
                merged.append((nxt_l, f"{r} {nxt_r}"))
                i += 1
            else:
                merged.append(('null', r))
        else:
            merged.append((l, r))
        i += 1

    return merged

def save_alignment_csv(s1: str, s2: str, filepath: str):
    """
    Aligns s1 and s2 and writes the result to a CSV with columns:
    left_token,right_token
    """
    pairs = align_hindi_sentences(s1, s2)
    with open(filepath, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['phrase1', 'phrase2'])
        writer.writerows(pairs)

# Example usage:
s1="मैम आप ओ पी डी में लेकर आ जाओ सुबह दस बजे और फर्स्ट टाइम में आ जाओगे तो आपको पेमेंट देना पड़ेगा"
s2="मैम आप ओ पी डी में ले के आ जाओ सुबह दस बजे और फर्स्ट टाइम आप आओगे तो आप को पेमेंट देना पड़ेगा"
save_alignment_csv(s1, s2, 'alignment.csv')
print("1")

s1="चार सौ रुपए बस ठीक लगेगी बात कुछ नहीं"
s2="चार सौ रुपये बस फीस लगेगी बाक़ी कुछ नहीं"
save_alignment_csv(s1, s2, 'alignment.csv')
print("2")

s1="मतलब उसमें क्या प्रॉब्लम है जैसे दूर का नहीं दिख ही रहा है या फिर नज़दीक का नहीं दिख ही रहा है"
s2="मतलब उस में क्या प्रोब्लम जैसे दूर का नहीं दिखाई दे रहा है या फिर नज़दीक का नहीं दिखाई दे"
save_alignment_csv(s1, s2, 'alignment.csv')
print("3")

s1="मोजाबिंदौर का क्या ऐज क्या है आपको"
s2="मोतियाबिंद हो रखा है क्या है ऐज क्या है आप"
save_alignment_csv(s1, s2, 'alignment.csv')
print("4")

s1="आपके पापा के ऐज क्या है"
s2="आपका पप्पा का एज क्या है"
save_alignment_csv(s1, s2, 'alignment.csv')
print("5")

s1="मैम आप अभी है ना इनको ज़्यादा मीठा खिलाना बंद कर दीजिए शुगर कंट्रोल करने के लिए और आप है ना एक बार आँखें आकर दिखा दीजिए उनको मोचपेन धोकर का है शायद तक क्या वो एज हो जाती है ना चालीस से ऊपर"
s2="मैम आप अभी है ना इन को ज़्यादा मीठा खिलाना बंद कर दीजिए शुगर कंट्रोल करने के लिए और आप है ना एक बार आँख आँखें दिखा दीजिए उन को मोतियाबिंद हो रखा है शायद क्या एज हो जाती है ना चालिस से ऊपर"
save_alignment_csv(s1, s2, 'alignment.csv')
print("6")

s1="हैलो हाँ जी आप आइसक्रीम पार्लर से बात कर रहे हैं सर"
s2="हैलो हाँ जी आप आइसक्रीम पार्लर से बात कर रहे है सर"
save_alignment_csv(s1, s2, 'alignment.csv')
print("7")

s1="यही है सर वो क्वालिटी में दिक्कत नहीं है कॉस्ट लगभग लगभग जैसे तीन सौ बच्चों के अपने क्या रह जाएगी हाँ एवरेज"
s2="ये सर बस क्वालिटी में दिक्कत नहीं कोस्ट लगभगलगभग जैसे तीन सौ बच्चों की अपने क्या रह जाएगी ऐवरेज"
save_alignment_csv(s1, s2, 'alignment.csv')
print("8")

s1="ये अपने सांगानेर थाने पे है ना यहाँ पे जा"
s2="ये अपने साहनेल थाड़े पे है ना यहाँ पे जो"
save_alignment_csv(s1, s2, 'alignment.csv')
print("9")

s1="क्रिमबेज चाहिए सर ओयल एक काम कीजिए आप दोनों के मेरे को दोदो तीनतीन सैम्पल भेज दीजिए"
s2="क्रीम बेस चाहिए सर ऑइल एक काम कीजिए आप दोनों के दोदो तीनतीन सैम्पल भेज दीजिए"
save_alignment_csv(s1, s2, 'alignment.csv')
print("10")

s1="तो अपने को एकएक स्टूडेंट विद्यार्थी को एक बटरस्कॉच और दो वनीले का पैकेट रखा है हमने एक तो अपने करीबन तीन सौ स्टूडेंट हैं तो उसके लिए अपने को सैम्पल के लिए बोला भाई अपने को एक बटरस्कॉच और दो वनीला का आप सैम्पल ऑयल वेस और क्रीम वाला दोनों बीजवा देते हैं"
s2="तो अपने को एकएक स्टूडेंट विद्यार्थी को एक बटरस्कॉच और दो वनीले का पेकेट रखा है अपन ने एक तो अपने तकरीबन तीन सौ स्टूडेंट हैं उसके लिए अपने को सैम्पल के लिए बोल भाई अपने को एक बटरस्कॉच और दो वनीला का आप सैम्पल ऑइल वेस और क्रीम वाला दोनों भिजवा देते"
save_alignment_csv(s1, s2, 'alignment.csv')
print("11")

s1="देखो हमारे यहाँ है बटरस्कॉच की है आपके अमेरिकन नट्स की है वनीला की है स्टोबेरी की है और हमारे पांगलोई की है कुल्फी है किस्ता है"
s2="देखो हमारे यहाँ है बटरस्कॉच की है आपके अमेरिकन नट्स की है वनिला की है स्ट्रॉबेरी की है और हमारे पान गिलोई की है कुल्फी है पिस्ता है"
save_alignment_csv(s1, s2, 'alignment.csv')
print("12")

s1="हाँ दूध वाले स्वाद की आइसक्रीम भी हैं"
s2="हाँ दूध वाले स्वाद के आइसक्रीम भी है"
save_alignment_csv(s1, s2, 'alignment.csv')
print("13")

s1="अच्छा एक ही और मैंने बताया ना आपको बड़ा वनीला का भी मिल जाएगा स्ट्रॉबेरी का भी मिल जाएगा आपने कहा है फ़ैमिली की फंक्शन के लिए आपको चाहिए तो कितने और फ़ैमिली फंक्शन में आपके कितने आदमी हैं"
s2="अच्छा एक ही और मैंने बताया न आपको बड़ा वनिला का भी मिल जाएगा स्ट्रॉबेरी का भी मिल जाएगा आपने कहा है फैमिली के फंक्शन के लिए आपको चाहिए तो कितने फैमिली फंक्शन में आपके कितने आदमी हैं"
save_alignment_csv(s1, s2, 'alignment.csv')
print("14")

s1="ठीक है मैं करवा दूँगा आपको कब चाहिए यह सब"
s2="ठीक है मैं करवा दूँगा आपको कब चाहिए यह सब"
save_alignment_csv(s1, s2, 'alignment.csv')
print("15")

s1="नहीं नहीं आपको मिल जाएगी आप मुझे बता दीजिए आप जो पेमेंट करेंगे वो कैसे करेंगे मुझे आधा पेमेंट तो पहले करना पड़ेगा आपको"
s2="नहींनहीं आपको मिल जाएगी आप मुझे बता दीजिये आप जो पेमेंट करेंगे वह कैसे करेंगे मुझे आधा पेमेंट तो पहले करना पड़ेगा आपको"
save_alignment_csv(s1, s2, 'alignment.csv')
print("16")

s1="फैमिली पैक के लिए हाँ दस मिल जाएंगे आप आर्डर बता दीजिए आपको कितने चाहिए उतने मैं मंगवा दूंगा आपको"
s2="फैमिली पैक के लिए हाँ दस मिल जाएंगे आप ऑर्डर बता दीजिए आपको कितने चाहिए उतने मैं मंगवा दूँगा आपको"
save_alignment_csv(s1, s2, 'alignment.csv')
print("17")


print("Saved alignment.csv")