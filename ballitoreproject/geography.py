from .imports import *
import diskcache as dc
geocache_obj = dc.Cache(os.path.join(PATH_DATA,'geocache'))

@cache
def get_nlp():
    import spacy
    nlp = spacy.load("en_core_web_sm")
    return nlp

@geocache_obj.memoize()
def get_named_places(text):
    # Process the text with spaCy
    nlp = get_nlp()
    doc = nlp(text)
    
    # Extract named entities labeled as GPE (Geopolitical Entity)
    places = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
    
    return places

@cache
def get_geolocator():
    from geopy.geocoders import Nominatim
    geolocator = Nominatim(user_agent="ballitoreproject")
    return geolocator

@geocache_obj.memoize()
def get_place_data(placename):
    loc = get_geolocator().geocode(placename)
    if not loc: return {}
    outd = {**loc.raw}
    outd['lat']=float(outd['lat'])
    outd['lon']=float(outd['lon'])
    return outd
