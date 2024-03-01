import nltk

nltk.download("stopwords")

from datetime import datetime, timezone
from nltk.corpus import stopwords
import re
from models.song_data import Song, Artist
import sqlalchemy

stop_words = stopwords.words("english")

epoch = datetime.utcfromtimestamp(0)


def now():
    return (
        datetime.now(timezone.utc) - datetime(1970, 1, 1, tzinfo=timezone.utc)
    ).total_seconds()


def iso_to_date(iso_date_str):
    return datetime.strptime(iso_date_str, "%Y-%m-%dT%H:%M:%S.%f")


def iso_to_unix(date):
    # ALWAYS MAKE SURE DATE OBJECT IS IN UTC
    try:
        utc_dt = datetime.strptime(str(date), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        utc_dt = datetime.strptime(str(date), "%Y-%m-%d %H:%M:%S.%f")
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


def strip_all_entities(text):
    return re.sub(r"@\w+|#\w+", "", text)


def strip_amp(text):
    stripped = text.replace("&amp;", "and")
    # Depending on if spaces have been put around certain punctuation
    stripped = stripped.replace("&amp ;", "and")
    return stripped


def remove_sc_and_numbers(text):
    return re.sub(r"[^\w\s]|\d", "", text)


def clean_text(text):
    return strip_all_entities(remove_contractions(strip_amp(text)))


def strip_stop_words(text):
    if text is None:
        return None
    precleaned_word_arr = text.split()
    cleaned_word_arr = [
        word.lower() for word in precleaned_word_arr if word.lower() not in stop_words
    ]
    return " ".join(cleaned_word_arr)


def save_track_info(track, session, hit):
    try:
        isrc = track["external_ids"]["isrc"]
    except KeyError:
        isrc = None

    existing_artist = (
        session.query(Artist.id).filter_by(spotify_id=track["artists"][0]["id"]).first()
    )

    artist_data = {
        "name": track["artists"][0]["name"],
        "spotify_id": track["artists"][0]["id"],
    }

    if existing_artist:
        artist_id = existing_artist.id
    else:
        new_artist = Artist()
        new_artist.update(artist_data)
        new_artist.save_to_db(session)
        artist_id = new_artist.id

    song_exists = session.query(
        sqlalchemy.exists().where(Song.spotify_id == track["id"])
    ).scalar()

    if not song_exists:
        new_song = Song(
            spotify_id=track["id"],
            isrc=isrc,
            artist_id=artist_id,
            title=track["name"],
            album=track["album"]["name"],
            year=track["album"]["release_date"][:4],
            explicit=track["explicit"],
            hit=hit,
            current_popularity=track["popularity"],
            non_hit_df_index=track.get("non_hit_df_index"),
        )
        new_song.save_to_db(session)
