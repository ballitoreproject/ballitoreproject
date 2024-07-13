from .imports import *


@cache
def get_metadata():
    o = []

    def fix_col(x):
        x = x.lower()
        if x == "internal id":
            return "id"
        # if 'date' in x: return 'date'
        if x == "date (mm/dd/yyyy)":
            return "date"
        if x == "date (yyyy)":
            return "year"
        if "unnamed" in x:
            return "notes"
        return x

    for fn in os.listdir(PATH_METADATA):
        if fn.endswith(".xlsx"):
            df = pd.read_excel(os.path.join(PATH_METADATA, fn))
            df.columns = [fix_col(c) for c in df]
            df = df[[c for c in df if c != "year"]]
            o.append(df)

    df = pd.concat(o).fillna("")
    df["id"] = df["id"].apply(lambda x: x.strip().lower().split(".txt")[0])
    df['box'] = df['id'].apply(extract_box_number)
    df["datetime"] = df["date"].apply(str).progress_apply(dateparser.parse)
    df["year"] = df["datetime"].apply(lambda x: x.year if x.year > 0 else 0)
    df["decade"] = df["year"].apply(lambda x: x // 10 * 10)
    df["is_journal"] = df.box.apply(lambda x: x in {13, 14})
    return df.set_index('id')


@cache
def get_txt_df():
    paths_txt = [
        os.path.join(root, fn)
        for root, dirs, fns in os.walk(PATH_TXT)
        for fn in fns
        if fn.endswith(".txt")
    ]

    o=[]
    for fnfn in tqdm(sorted(paths_txt), desc="Reading txt files"):
        with open(fnfn) as f:
            txt = f.read()
        d={
            'id':os.path.basename(fnfn).split(".txt")[0].lower(),
            'txt': txt.strip()
        }
        o.append(d)
    return pd.DataFrame(o).set_index('id')


def get_data(force=False):
    if not force and os.path.exists(PATH_COMBINED):
        return pd.read_excel(PATH_COMBINED).set_index("id").fillna('')

    odf = get_metadata().join(get_txt_df(),how='outer').fillna('')
    odf["year"] = odf["datetime"].apply(lambda x: x.year if x.year > 0 else 0)
    odf["decade"] = odf["year"].apply(lambda x: x // 10 * 10)

    odf["box"] = [extract_box_number(x) for x in odf.index]
    odf["is_journal"] = odf.box.apply(lambda x: x in {13, 14})
    odf = merge_letters_across_pages(odf)
    odf = odf.sort_values(["box", "datetime"])
    odf['num_letters']=1
    odf['num_words']=odf['txt'].progress_apply(lambda x: len(tokenize(x)))
    odf.to_excel(PATH_COMBINED)
    return odf

@cache
def cached_ballitore_data():
    return get_data()

def get_text(id):
    try:
        return get_txt_df().loc[id].txt
    except Exception:
        return ''

def merge_letters_across_pages(odf):
    # @title Merging letters across pages
    last_id = None
    id2group = defaultdict(list)
    for id, row in odf.iterrows():
        if row.notes.strip().startswith("p.") and not id.endswith("001"):
            id2group[last_id] += [{"id": id, "notes": row.notes, "txt": row.txt}]
        else:
            last_id = id

    duplicated = {d["id"] for ld in id2group.values() for d in ld}
    odf2 = odf[~odf.index.isin(duplicated)].copy()
    id2txt = dict(zip(odf2.index, odf2.txt))
    id2notes = dict(zip(odf2.index, odf2.notes))
    id2ids = defaultdict(list)
    for id, ld in id2group.items():
        for d in sorted(ld, key=lambda x: x["id"]):
            id2txt[id] += d["txt"]
            id2notes[id] += "; " + d["notes"]
            id2ids[id] += [d["id"]]

    odf2["notes"] = odf2.index.map(id2notes)
    odf2["txt"] = odf2.index.map(id2txt)
    odf2["supplemental_ids"] = odf2.index.map(id2ids)
    odf2["supplemental_ids"] = odf2["supplemental_ids"].apply(lambda x: "; ".join(x))
    odf2[odf2.supplemental_ids.apply(len) > 0]
    return odf2
