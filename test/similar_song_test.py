from feature.similar_songs import similar_songs


input = 'audio'

recomendator = similar_songs(input)

print(f' output: {recomendator.main()}')