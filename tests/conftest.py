
from pathlib import Path

import pytest

from iantirta.audio import AudioSegment
from iantirta.models import TextTiming, WordTiming


@pytest.fixture
def separated_path() -> Path:
    return Path(__file__).parent.parent / "tests" / 'separated.wav'


@pytest.fixture
def indo_lyric() -> str:
    return """Belum ada satu bulan
Ku yakin masih ada sisa wangiku di bajumu
Namun, kau tampak baik saja
Bahkan senyummu lebih lepas
Sedang aku di sini hampir gila
Kita tak temukan jalan
Sepakat akhiri setelah beribu debat panjang
Namun, kau tampak baik saja
Bahkan senyummu lebih lepas
Sedang aku di sini belum terima
Bohongkah tangismu sore itu di pelukku?
Nyatanya pergiku pun tak lagi mengganggumu
Apa sudah ada kabar lain yang kautunggu?
Sudah adakah yang gantikanku?
Yang khawatirkanmu setiap waktu
Yang cerita tentang apa pun sampai hal-hal tak perlu
Kalau bisa, jangan buru-buru
Kalau bisa, jangan ada dulu
Baru lewat satu bulan
Kemarin ulang tahunku, tak ada pesan darimu
Tak apa, mungkin kau lupa
Atau sudah ada hati yang harus kaujaga
Sudah adakah yang gantikanku?
Yang kauantar jemput setiap Sabtu
Yang s'lalu ingatkan untuk pakai sabuk pengamanmu
Kalau bisa, jangan buru-buru
Sudah adakah yang gantikanku?
Yang khawatirkanmu setiap waktu
Yang cerita tentang apa pun sampai hal-hal tak perlu
Kalau bisa, jangan buru-buru
Kalau bisa, jangan ada dulu
Hu, hu-uh
Hu-uh
"""


@pytest.fixture
def mandarin_lyric() -> str:
    return """
慌啥子嘛，慢慢慢来  
太阳下山，有月亮爬上来  
云会散，花会开  
好运总会为你而来  

卸下一身的疲惫  
今夜枕着星光入睡  
风儿轻轻吹  
这一刻就忘了你是谁  

又是清醒吹又是醉  
别把所有压力自己背  
人生路还远  
你不必事事都完美  

慌啥子嘛，慢慢慢来  
太阳下山，有月亮爬上来  
云会散，花会开  
好运总会为你而来  

慌啥子嘛，慢慢慢来  
每个人都有自己的节拍  
得不到，就放开  
一切都是最好安排  

卸下一身的疲惫  
今夜枕着星光入睡  
风儿轻轻吹  
这一刻就忘了你是谁  

又是清醒吹又是醉  
别把所有压力自己背  
人生路还远  
你不必事事都完美  

慌啥子嘛，慢慢慢来  
太阳下山，有月亮爬上来  
云会散，花会开  
好运总会为你而来  

慌啥子嘛，慢慢慢来  
每个人都有自己的节拍  
得不到，就放开  
一切都是最好安排
"""


@pytest.fixture
def audio_segments() -> list[AudioSegment]:
    return [
        AudioSegment(start=5.632, end=8.336),
        AudioSegment(start=8.352, end=16.368),
        AudioSegment(start=16.384, end=20.288),
        AudioSegment(start=20.528, end=25.824),
        AudioSegment(start=26.464, end=30.464),
        AudioSegment(start=33.344, end=38.224),
        AudioSegment(start=38.848, end=45.088),
        AudioSegment(start=45.328, end=50.704),
        AudioSegment(start=51.344, end=58.88),
        AudioSegment(start=59.088, end=62.56),
        AudioSegment(start=62.848, end=68.096),
        AudioSegment(start=68.496, end=73.888),
        AudioSegment(start=73.904, end=79.232),
        AudioSegment(start=79.392, end=85.088),
        AudioSegment(start=85.104, end=90.24),
        AudioSegment(start=90.416, end=95.04),
        AudioSegment(start=95.952, end=101.424),
        AudioSegment(start=106.432, end=109.6),
        AudioSegment(start=109.76, end=115.456),
        AudioSegment(start=115.84, end=121.248),
        AudioSegment(start=121.504, end=131.392),
        AudioSegment(start=131.552, end=137.296),
        AudioSegment(start=137.312, end=141.872),
        AudioSegment(start=142.208, end=146.0),
        AudioSegment(start=146.016, end=153.312),
        AudioSegment(start=153.6, end=161.12),
        AudioSegment(start=161.136, end=164.512),
        AudioSegment(start=164.896, end=167.616),
        AudioSegment(start=167.632, end=175.2),
        AudioSegment(start=175.44, end=181.808),
        AudioSegment(start=187.44, end=189.792),
        AudioSegment(start=189.808, end=195.328),
        AudioSegment(start=195.344, end=200.816),
        AudioSegment(start=202.528, end=211.184),
        AudioSegment(start=211.392, end=216.464),
        AudioSegment(start=216.48, end=221.6),
        AudioSegment(start=221.616, end=222.832),
        AudioSegment(start=222.848, end=223.936),
        AudioSegment(start=223.952, end=228.4),
    ]


@pytest.fixture
def transcribe_res():
    return [
        TextTiming(words=[
            WordTiming(start=None, end=None, word='Kini', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='serahkan', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='semua', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='ujungnya', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='pada', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='takdir', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='Jika', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='beruntung,', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='maka', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='kisah', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='ini', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='tak', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='akan', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='berakhir', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='Untuk', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='sejenak', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='kumohon', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='berhenti', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='perdebatkan', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='yang', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='tak', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='penting', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='Kita', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='berdamai,', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='berjabat', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='tangan,', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='tertawakan', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='yang', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='tak', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='bisa', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='dibenahi', score=None, language=None, _clean=None)
        ], language='id', ass_event=None, _clean=None, _text=None),
        TextTiming(words=[
            WordTiming(start=None, end=None, word='Hanya', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='pada', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='takdir.', score=None, language=None, _clean=None)
        ], language='id', ass_event=None, _clean=None, _text=None),
        TextTiming(words=[
            WordTiming(start=None, end=None, word='Jika', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='warung', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='tunggak', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='kaki', score=None, language=None, _clean=None)
        ], language='id', ass_event=None, _clean=None, _text=None),
        TextTiming(words=[
            WordTiming(start=None, end=None, word='Tak', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='akan', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='berakhir', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='untuk', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='selamanya.', score=None, language=None, _clean=None)
        ], language='id', ass_event=None, _clean=None, _text=None),
        TextTiming(words=[
            WordTiming(start=None, end=None, word='Mohon', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='berhenti', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='pada', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='bahkan', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='yang', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='tak', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='penting', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='kita', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='berdamai', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='berjabat', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='tangan.', score=None, language=None, _clean=None)
        ], language='id', ass_event=None, _clean=None, _text=None),
        TextTiming(words=[
            WordTiming(start=None, end=None, word='Banyak.', score=None, language=None, _clean=None)
        ], language='id', ass_event=None, _clean=None, _text=None), 
        TextTiming(words=[
            WordTiming(start=None, end=None, word='anggap', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='saja.', score=None, language=None, _clean=None)
        ], language='id', ass_event=None, _clean=None, _text=None),
        TextTiming(words=[
            WordTiming(start=None, end=None, word='Jabat', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='tangan.', score=None, language=None, _clean=None)
        ], language='id', ass_event=None, _clean=None, _text=None),
        TextTiming(words=[
            WordTiming(start=None, end=None, word='cerita', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='seperti', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='apa', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='yang', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='mau', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='kau', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='kenang?', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='Kita', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='buat', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='malam', score=None, language=None, _clean=None)
        ], language='id', ass_event=None, _clean=None, _text=None),
        TextTiming(words=[
            WordTiming(start=None, end=None, word='Jadi', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='saat', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='waktunya', score=None, language=None, _clean=None)
        ], language='id', ass_event=None, _clean=None, _text=None),
        TextTiming(words=[
            WordTiming(start=None, end=None, word='Jika', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='ada', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='yang', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='salah,', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='jika', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='kau', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='kecewa,', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='tolong', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='jagalah.', score=None, language=None, _clean=None)
        ], language='id', ass_event=None, _clean=None, _text=None),
        TextTiming(words=[
            WordTiming(start=None, end=None, word='Kita', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='buat', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='malam', score=None, language=None, _clean=None)
        ], language='id', ass_event=None, _clean=None, _text=None),
        TextTiming(words=[
            WordTiming(start=None, end=None, word='Bahkan', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='tahu.', score=None, language=None, _clean=None)
        ], language='id', ass_event=None, _clean=None, _text=None),
        TextTiming(words=[
            WordTiming(start=None, end=None, word='Zagi', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='kau', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='masih', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='bisa.', score=None, language=None, _clean=None)
        ], language='id', ass_event=None, _clean=None, _text=None), 
        TextTiming(words=[WordTiming(start=None, end=None, word='Kita', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='buat', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='menenangkan.', score=None, language=None, _clean=None)
        ], language='id', ass_event=None, _clean=None, _text=None),
        TextTiming(words=[
            WordTiming(start=None, end=None, word='Ah,', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='ah,', score=None, language=None, _clean=None)
        ], language='id', ass_event=None, _clean=None, _text=None),
        TextTiming(words=[
            WordTiming(start=None, end=None, word='Tiara', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='yang', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='kau', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='pasi,', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='mungkin', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='dalam', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='hit', score=None, language=None, _clean=None)
        ], language='id', ass_event=None, _clean=None, _text=None),
        TextTiming(words=[
            WordTiming(start=None, end=None, word='Dan', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='bisa', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='jadi', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='kita', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='tak.', score=None, language=None, _clean=None)
        ], language='id', ass_event=None, _clean=None, _text=None),
        TextTiming(words=[
            WordTiming(start=None, end=None, word='Jangan', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='takut,', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='jangan', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='takut,', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='jangan', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='takut.', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='Jangan', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='takut,', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='jangan', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='takut,', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='jangan', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='takut.', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='Jangan', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='takut,', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='jangan', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='takut,', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='jangan', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='takut.', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='Jangan', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='takut,', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='jangan', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='takut,', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='jangan', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='takut.', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='Jangan', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='takut,', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='jangan', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='takut,', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='jangan', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='takut.', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='Jangan', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='takut,', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='jangan', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='takut,', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='jangan', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='takut.', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='Jangan', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='takut,', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='jangan', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='takut,', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='jangan', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='takut.', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='Jangan', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='takut,', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='jangan', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='takut,', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='jangan', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='takut.', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='Jangan', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='takut,', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='jangan', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='takut,', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='jangan', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='takut.', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='Jangan', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='takut,', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='jangan', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='takut,', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='jangan', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='takut.', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='Jangan', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='takut,', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='jangan', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='takut,', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='jangan', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='takut.', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='Jangan', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='takut,', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='jangan', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='takut,', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='jangan', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='takut.', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='Jangan', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='takut,', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='jangan', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='takut,', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='jangan', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='takut.', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='Jangan', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='takut,', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='jangan', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='takut,', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='jangan', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='takut.', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='Jangan', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='takut,', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='jangan', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='takut,', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='jangan', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='takut.', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='Jangan', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='takut,', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='jangan', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='takut,', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='jangan', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='takut.', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='Jangan', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='takut,', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='jangan', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='takut,', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='jangan', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='takut.', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='J', score=None, language=None, _clean=None)
        ], language='id', ass_event=None, _clean=None, _text=None),
        TextTiming(words=[
            WordTiming(start=None, end=None, word='Terima', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='kasih', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='karena', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='mengenalmu.', score=None, language=None, _clean=None)
        ], language='id', ass_event=None, _clean=None, _text=None),
        TextTiming(words=[
            WordTiming(start=None, end=None, word='Aku', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='yang', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='seperti', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='ini.', score=None, language=None, _clean=None)
        ], language='id', ass_event=None, _clean=None, _text=None),
        TextTiming(words=[
            WordTiming(start=None, end=None, word='Anggap', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='saja', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='besok', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='ini', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='semuanya,', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='semoga', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='yang', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='kau', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='ingat', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='hanya.', score=None, language=None, _clean=None)
        ], language='id', ass_event=None, _clean=None, _text=None),
        TextTiming(words=[
            WordTiming(start=None, end=None, word='Kita', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='buat', score=None, language=None, _clean=None)
        ], language='id', ass_event=None, _clean=None, _text=None),
        TextTiming(words=[
            WordTiming(start=None, end=None, word='This', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='is', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='how', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='I', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='do', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='it.', score=None, language=None, _clean=None)
        ], language='id', ass_event=None, _clean=None, _text=None),
        TextTiming(words=[
            WordTiming(start=None, end=None, word='Jika', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='ada', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='yang', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='salah,', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='jika', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='kau', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='kecewa.', score=None, language=None, _clean=None)
        ], language='id', ass_event=None, _clean=None, _text=None),
        TextTiming(words=[
            WordTiming(start=None, end=None, word='Makan,', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='kita', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='buat', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='makan.', score=None, language=None, _clean=None)
        ], language='id', ass_event=None, _clean=None, _text=None),
        TextTiming(words=[
            WordTiming(start=None, end=None, word='Tak', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='akan', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='tahu', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='sampai', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='kapan.', score=None, language=None, _clean=None)
        ], language='id', ass_event=None, _clean=None, _text=None),
        TextTiming(words=[
            WordTiming(start=None, end=None, word='Kita', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='buat', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='dunia', score=None, language=None, _clean=None)
        ], language='id', ass_event=None, _clean=None, _text=None),
        TextTiming(words=[
            WordTiming(start=None, end=None, word='Aku', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='takkan', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='pernah', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='menyerah.', score=None, language=None, _clean=None)
        ], language='id', ass_event=None, _clean=None, _text=None),
        TextTiming(words=[
            WordTiming(start=None, end=None, word='Oke.', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='Oke.Oke', score=None, language=None, _clean=None)
        ], language='id', ass_event=None, _clean=None, _text=None),
        TextTiming(words=[
            WordTiming(start=None, end=None, word='Oke.', score=None, language=None, _clean=None)
        ], language='id', ass_event=None, _clean=None, _text=None),
        TextTiming(words=[
            WordTiming(start=None, end=None, word='Aku', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='takkan', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='pernah', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='menyerah.', score=None, language=None, _clean=None)
        ], language='id', ass_event=None, _clean=None, _text=None),
        TextTiming(words=[
            WordTiming(start=None, end=None, word='Aku', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='takkan', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='pernah', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='menyerah.', score=None, language=None, _clean=None)
        ], language='id', ass_event=None, _clean=None, _text=None),
        TextTiming(words=[
            WordTiming(start=None, end=None, word='Kita', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='buat', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='menyanyi.', score=None, language=None, _clean=None)
        ], language='id', ass_event=None, _clean=None, _text=None),
        TextTiming(words=[
            WordTiming(start=None, end=None, word='Angka', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='desa', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='satu', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='yang', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='ada.', score=None, language=None, _clean=None)
        ], language='id', ass_event=None, _clean=None, _text=None),
        TextTiming(words=[
            WordTiming(start=None, end=None, word='Jika', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='ada', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='yang', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='salah,', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='jika', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='kau', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='kecewa,', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='tolong', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='maaf.', score=None, language=None, _clean=None)
        ], language='id', ass_event=None, _clean=None, _text=None),
        TextTiming(words=[
            WordTiming(start=None, end=None, word='Kita', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='buat', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='manusia.', score=None, language=None, _clean=None)
        ], language='id', ass_event=None, _clean=None, _text=None),
        TextTiming(words=[
            WordTiming(start=None, end=None, word='Takkan', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='tahu', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='sampai', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='kapan,', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='selagi', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='kau', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='masih', score=None, language=None, _clean=None)
        ], language='id', ass_event=None, _clean=None, _text=None),
        TextTiming(words=[
            WordTiming(start=None, end=None, word='Di', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='bintang.', score=None, language=None, _clean=None)
        ], language='id', ass_event=None, _clean=None, _text=None),
        TextTiming(words=[
            WordTiming(start=None, end=None, word='Walaupun', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='nangka', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='selagi', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='kau', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='masih', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='bisa.', score=None, language=None, _clean=None)
        ], language='id', ass_event=None, _clean=None, _text=None),
        TextTiming(words=[
            WordTiming(start=None, end=None, word='Kita', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='buat', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='menangga.', score=None, language=None, _clean=None)
        ], language='id', ass_event=None, _clean=None, _text=None),
        TextTiming(words=[
            WordTiming(start=None, end=None, word='Aku', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='takkan', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='pernah', score=None, language=None, _clean=None),
            WordTiming(start=None, end=None, word='menyerah.', score=None, language=None, _clean=None)
        ], language='id', ass_event=None, _clean=None, _text=None)] 
