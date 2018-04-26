import re
import collections
import string
from nltk.util import ngrams
from misc.convert import Convert
import os
'''
calculate the number of n-gram overlapping between context and entity description
'''
class Candidate(object):
    def __init__(self, title, wikiid):
        self.title = title
        self.wikiid = wikiid
        self.dscpt = ""

class Mention(object):
    def __init__(self, text, offset, candid, answer, local_context):
        self.text = text
        self.offset = offset
        self.candid = candid
        self.answer = answer
        self.local_context = local_context

def get_n_gram(text, n):
    #get rid of useless notations
    text = re.sub("<.*>", "", text)
    # get rid of punctuation (except periods!)
    punctuationNoPeriod = "[" + re.sub("\.", "", string.punctuation) + "]"
    text = re.sub(punctuationNoPeriod, "", text)
    tokens = text.split()
    n_gram = ngrams(tokens, n)
    n_gram_freq = collections.Counter(n_gram)

    return n_gram_freq

#input: two counters
def compare_n_gram(ctx, dscpt):
    overlap_num = 0
    overlap_n_gram = []

    for k, count in ctx.items():
        if k in dscpt.keys():
            dscpt_count = dscpt[k]
            overlap_num += min(count, dscpt_count)
            overlap_n_gram.append(k)

    return overlap_num, overlap_n_gram

def get_part_dscpt(dscpt, window):
    part_dscpt = ""
    st = window[0]
    ed = window[1]
    tokens = dscpt.strip().split()

    if st > len(tokens):
        st = 0
        ed = len(tokens)
    else:
        if ed > len(tokens):
            ed = len(tokens)
    part_dscpt = " ".join(tokens[st:ed])

    return part_dscpt

def get_local_ctx(ctx, mention, offset, ctx_window):
    #add special notation
    st = offset
    ed = offset + len(mention)
    ctx = ctx[:ed] + "]" + ctx[ed:]
    ctx = ctx[:offset] + "[" + ctx[st:]

    st = ctx.index("[")
    ed = ctx.index("]")

    if ctx[st + 1:ed] != mention:
        print(ctx[st + 1:ed], mention)
        return ctx

    st = None
    ed = None
    tokens = ctx.split()
    for i in range(len(tokens)):
        t = tokens[i]
        if "[" in t:
            st = i
            tokens[i] = tokens[i].replace("[", "")
        if "]" in t:
            ed = i
            tokens[i] = tokens[i].replace("]", "")
    assert st != None and ed != None

    local_ctx = " ".join(tokens[max(0, st -  ctx_window): min(len(tokens), ed + ctx_window)])

    return local_ctx



'''
ctx_window: the window size among mention
n: n-gram
dscpt_window: the range of description [A, B]
data_path: query, answer path
'''
def main(ctx_window, dscpt_window,
         dscpt_path, data_path, mention_candid_map_path,
         doc_id_list_path, n, save_path, wiki_subset_path):

    activate_entity = {} # entities that have been triggered in this dataset
    #load doc id
    doc_id_list = []
    with open(doc_id_list_path, "r", encoding="utf-8") as f:
        for line in f:
            doc_id = line.replace("\n", "")
            doc_id_list.append(doc_id)
    print("[INFO] get all document id")

    #load mapping
    mention_candid_map = {}
    with open(mention_candid_map_path, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split("\t")
            mention = tokens[0]
            candid = tokens[1].split("|")
            mention_candid_map[mention] = candid
    print("[INFO] completed loading mention candidate map")

    #load description
    entity_dscpt_map = {}
    with open(dscpt_path, "r", encoding = "utf-8") as f:
        for line in f:
            tokens = line.strip().split("\t")
            if len(tokens) != 2:
                print("Line tokens != 2", tokens)
                continue
            entity, dscpt = tokens
            entity_dscpt_map[entity] = get_part_dscpt(dscpt, dscpt_window)
    print("[INFO] completed loading candidate description")

    for ng in n:
        cur_save_path = save_path + str(ng) + ".tsv"
        fsave = open(cur_save_path, "w+", encoding="utf-8")
        for doc_id in doc_id_list:
            context = ""
            with open(os.path.join(data_path, "query", doc_id), "r", encoding="utf-8") as f:
                context = f.read()
            #get mentions, their local context, candidate list
            with open(os.path.join(data_path, "answer", doc_id), "r", encoding="utf-8") as f:
                mention_list = []
                for line in f:
                    mention, offset, _, answer = line.strip().split("\t")
                    offset = int(offset)
                    if mention in mention_candid_map.keys():
                        candidates = mention_candid_map[mention]
                    else:
                        continue
                    local_context = get_local_ctx(context, mention, offset, ctx_window)
                    mention_list.append(Mention(mention, offset, candidates, answer, local_context))

            for mention in mention_list:
                context_n_gram = get_n_gram(mention.local_context.lower(), ng)
                gold_answer = mention.answer
                for candid in mention.candid:
                    #convert to a special string
                    candid_name = Convert().convert_to_title(candid)
                    if candid_name in entity_dscpt_map.keys():
                        candid_dscpt = entity_dscpt_map[candid_name]
                        dscpt_n_gram = get_n_gram(candid_dscpt.lower(), ng)
                        overlap_num, overlap_n_gram = compare_n_gram(context_n_gram, dscpt_n_gram)
                        fsave.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(doc_id, mention.text, str(mention.offset),
                                                                  gold_answer, candid, str(overlap_num), str(overlap_n_gram)))
                        activate_entity[candid_name] = entity_dscpt_map[candid_name]
        fsave.close()

    with open(wiki_subset_path, "w+",encoding = "utf-8") as f:
        print("[INFO] number of triggered candidate entities:{}".format(len(activate_entity.keys())))
        for k, v in activate_entity.items():
            f.write("{}\t{}\n".format(k, v))



if __name__ == "__main__":
    main(50, [0, 600], "./data/wiki_txt.tsv", "./data/evaluation/conll2003_train/",
         "./data/aida_mapping.tsv", "./data/evaluation/conll2003_train/doc_id_list.tsv",
         [1,2,3], "./data/result/30_0_600_lower/conll2003_train_ngram_","./data/wiki_txt_conll2003.tsv")

