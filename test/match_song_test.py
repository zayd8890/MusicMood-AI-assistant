from feature.match_song import match_song


input = 'audio'

metadata = match_song(input)

print(f' output: {metadata.main()}')