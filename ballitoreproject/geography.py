from .imports import *

@cache
def get_nlp():
    import spacy
    nlp = spacy.load("en_core_web_sm")
    return nlp

@sqlitedict_cache(PATH_NER_DATA)
def get_ner_data_for_id(id):
    from .ballitoreproject import get_text
    nlp = get_nlp()
    text = get_text(id)
    doc = nlp(text)
    return [{'ent':ent.text, 'ent_type':ent.label_} for ent in doc.ents]

def get_named_places_for_id(id, ent_type='GPE'):
    return [d['ent'] for d in get_ner_data_for_id(id) if d['ent_type'] == ent_type]

def get_named_people_for_id(id, ent_type='PERSON'):
    return [d['ent'] for d in get_ner_data_for_id(id) if d['ent_type'] == ent_type]


@cache
def get_geolocator():
    from geopy.geocoders import Nominatim
    geolocator = Nominatim(user_agent="ballitore")
    return geolocator

@sqlitedict_cache(PATH_GEOLOC_DATA)
def get_place_data(placename):
    loc = get_geolocator().geocode(placename)
    if not loc: return {}
    outd = {**loc.raw}
    outd['lat']=float(outd['lat'])
    outd['lon']=float(outd['lon'])
    return outd
