import sqlalchemy as db, ForeignKey
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class BaseModel(Base):
    # Provides common helper methods to models

    def update(self, newdata):
        # Updates model attributes based on the keys present in a dictionary
        for key, value in newdata.items():
            setattr(self, key, value)

    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

    def save_to_db(self):
        db.session.add(self)
        db.session.commit()

    def delete(self):
        db.session.delete(self)
        db.session.commit()


class Song(BaseModel):
    __tablename__ = 'song'

    id = db.Column(db.Integer(), primary_key=True)
    artist_id = db.Column(db.Integer, ForeignKey('artist.id'))

    song = db.Column(db.Text())
    spotify_id = db.Column(db.Text())
    uri = db.Column(db.Text())
    isrc = db.Column(db.Text())
    album = db.Column(db.Text())
    analysis_url = db.Column(db.Text())
    track_href = db.Column(db.Text())
    type = db.Column(db.Text())
    current_popularity = db.Column(db.Float())
    year = db.Column(db.Integer())
    explicit = db.Column(db.Boolean())
    hit = db.Column(db.Boolean())

    danceability = db.Column(db.Float())
    energy = db.Column(db.Float())
    key = db.Column(db.Integer())
    mode = db.Column(db.Integer())
    speechiness = db.Column(db.Float())
    acousticness = db.Column(db.Float())
    instrumentalness = db.Column(db.Float())
    liveness = db.Column(db.Float())
    valence = db.Column(db.Float())
    tempo = db.Column(db.Float())
    duration_ms = db.Column(db.Float())
    time_signature = db.Column(db.Integer())

    lyrics = db.Column(db.Text())
    polarity = db.Column(db.Float())
    subjectivity = db.Column(db.Float())
    lyric_length = db.Column(db.Integer())


class Artist(BaseModel):
    __tablename__ = 'artist'

    id = db.Column(db.Integer(), primary_key=True)
    name = db.Column(db.Text())
    artist_id = db.Column(db.Text())
    artist_popularity = db.Column(db.Float())
    artist_num_hits = db.Column(db.Integer())

    song = relationship("Song", back_populates="artist")
