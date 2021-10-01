from datetime import datetime, timezone
from nltk.corpus import stopwords
import re
import demoji
import numpy as np
from tensorflow.keras import backend as K
import boto3
import base64
from models.song_data import Song, Artist

stop_words = stopwords.words('english')

epoch = datetime.utcfromtimestamp(0)


def now():
    return (datetime.now(timezone.utc) - datetime(1970, 1, 1, tzinfo=timezone.utc)).total_seconds()


def iso_to_date(iso_date_str):
    return datetime.strptime(iso_date_str, "%Y-%m-%dT%H:%M:%S.%f")


def iso_to_unix(date):
    # ALWAYS MAKE SURE DATE OBJECT IS IN UTC
    try:
        utc_dt = datetime.strptime(str(date), '%Y-%m-%d %H:%M:%S')
    except ValueError:
        utc_dt = datetime.strptime(str(date), '%Y-%m-%d %H:%M:%S.%f')
    timestamp = (utc_dt - epoch).total_seconds()
    return timestamp


def unix_to_iso(timestamp):
    # Timestamp in seconds
    return datetime.utcfromtimestamp(timestamp).replace(tzinfo=timezone.utc).isoformat()


def age_from_unix(timestamp):
    return now() - timestamp


def get_sentiment(blob):
    return blob.sentiment.polarity, blob.sentiment.subjectivity


def remove_contractions(text):
    text = text.lower()
    text = text.replace("i'm", "i am")
    text = text.replace("i'd", "i had")
    text = text.replace("i've", "i have")
    text = text.replace("you're", "you are")
    text = text.replace("he's", "he is")
    text = text.replace("he'd", "he would")
    text = text.replace("she's", "she is")
    text = text.replace("she'd", "she would")
    text = text.replace("it's", "it is")
    text = text.replace("we're", "we are")
    text = text.replace("we've", "we have")
    text = text.replace("they're", "they are")
    text = text.replace("they've", "they have")
    text = text.replace("they'd", "they would")
    text = text.replace("i'll", "i will")
    text = text.replace("you'll", "you will")
    text = text.replace("you've", "you have")
    text = text.replace("he'll", "he will")
    text = text.replace("she'll", "she will")
    text = text.replace("it'll", "it will")
    text = text.replace("we'll", "we will")
    text = text.replace("they'll", "they will")
    text = text.replace("isn't", "is not")
    text = text.replace("aren't", "are not")
    text = text.replace("won't", "will not")
    text = text.replace("don't", "do not")
    text = text.replace("didn't", "did not")
    text = text.replace("doesn't", "does not")
    text = text.replace("can't", "cannot")
    text = text.replace("couldn't", "could not")
    text = text.replace("wouldn't", "would not")
    text = text.replace("shouldn't", "should not")
    text = text.replace("hadn't", "had not")
    text = text.replace("hasn't", "has not")
    text = text.replace("haven't", "have not")
    text = text.replace("let's", "let us")
    text = text.replace("mightn't", "might not")
    text = text.replace("mustn't", "must not")
    text = text.replace("shan't", "shall not")
    text = text.replace("weren't", "were not")
    text = text.replace("that's", "that is")
    text = text.replace("there's", "there is")
    text = text.replace("we'd", "we would")
    text = text.replace("what'll", "what will")
    text = text.replace("what're", "what are")
    text = text.replace("what's", "what is")
    text = text.replace("what've", "what have")
    text = text.replace("where's", "where is")
    text = text.replace("who's", "who is")
    text = text.replace("who'd", "who would")
    text = text.replace("who're", "who are")
    text = text.replace("who'll", "who will")
    text = text.replace("who've", "who have")
    text = text.replace("you'd", "you would")
    text = text.replace("whomst'd've", "whomst did")
    text = text.replace("'s", "")
    return text


def remove_emojis(text):
    return demoji.replace(text)


def get_links(text):
    link_regex = re.compile(
        '((((https?):((//)|(\\\\))+)|pic\.twitter\.com\/)([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)',
        re.DOTALL
    )
    links = re.findall(link_regex, text)
    return links


def get_hashtags(text):
    return re.findall('#\w+', text)


def get_mentions(text):
    return re.findall('@\w+', text)


def strip_links(text):
    links = get_links(text)
    for link in links:
        text = text.replace(link[0], '')
    return text


def strip_all_entities(text):
    return re.sub(r'@\w+|#\w+', '', text)


def strip_amp(text):
    stripped = text.replace('&amp;', 'and')
    # Depending on if spaces have been put around certain punctuation
    stripped = stripped.replace('&amp ;', 'and')
    return stripped


def remove_sc_and_numbers(text):
    return re.sub(r'[^\w\s]|\d', '', text)


def clean_text(text):
    return strip_all_entities(remove_contractions(strip_links(remove_emojis(strip_amp(text)))))


def strip_stop_words(text):
    if text is None:
        return None
    precleaned_word_arr = text.split()
    cleaned_word_arr = [word.lower()
                        for word in precleaned_word_arr if word.lower() not in stop_words]
    return ' '.join(cleaned_word_arr)


def ioa(o, s):
    """
    Uses MAE instead of MSE
    input:
        o: observed
        s: simulated
    """
    try:
        ia = 1 - (K.sum(K.abs(o-s)))/(K.sum(K.abs(s-K.mean(o))+K.abs(o-K.mean(o))))
    except:
        ia = 1 - (np.sum(np.abs(o-s)))/(np.sum(np.abs(s-np.mean(o))+np.abs(o-np.mean(o))))
    return ia


def cd(y_true, y_pred):
    SS_res = np.sum(np.square(y_true - y_pred))
    SS_tot = np.sum(np.square(y_true - np.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))


def adj_r2(y_true, y_pred, n, p):
    return 1-(1-cd(y_true, y_pred))*(n-1)/(n-p-1)


def get_aws_secret(secret_name, region_name):
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    get_secret_value_response = client.get_secret_value(
        SecretId=secret_name
    )

    if 'SecretString' in get_secret_value_response:
        secret = get_secret_value_response['SecretString']
    else:
        secret = base64.b64decode(get_secret_value_response['SecretBinary'])

    return secret


def save_track_info(track, session, hit):
    try:
        isrc = track['external_ids']['isrc']
    except KeyError:
        isrc = None

    existing_artist = session.query(Artist.id).filter_by(spotify_id=track['artists'][0]['id']).first()
        
    artist_data = {
        'name': track['artists'][0]['name'],
        'spotify_id': track['artists'][0]['id']
    }

    if existing_artist:
        artist_id = existing_artist.id
    else:
        new_artist = Artist()
        new_artist.update(artist_data)
        new_artist.save_to_db(session)
        artist_id = new_artist.id

    song_exists = session.query(Song.id).filter_by(spotify_id=track['id']).first() is not None

    if 'non_hit_df_index' in track:
        non_hit_df_index = track['non_hit_df_index']
    else:
        non_hit_df_index = None

    song_data = {
        'spotify_id': track['id'],
        'isrc': isrc,
        'artist_id': artist_id,
        'title': track['name'],
        'album': track['album']['name'],
        'year': track['album']['release_date'][:4],
        'explicit': track['explicit'],
        'hit': hit,
        'current_popularity': track['popularity'],
        'non_hit_df_index': non_hit_df_index
    }

    if not song_exists:
        new_song = Song()
        new_song.update(song_data)
        new_song.save_to_db(session)