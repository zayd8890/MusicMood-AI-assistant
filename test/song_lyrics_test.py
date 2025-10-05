from feature.song_lyrics import song_lyrics


input = 'audio'

song = song_lyrics(input)

print(f' output: {song.main()}')