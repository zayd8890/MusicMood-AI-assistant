from feature.song_generation import song_generation


input = 'audio'

generator = song_generation(input)

print(f' output: {generator.main()}')