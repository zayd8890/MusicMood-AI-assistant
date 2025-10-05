from feature.mood_recommendation import mood_recommendation


input = 'text'

recomendator = mood_recommendation(input)

print(f' output: {recomendator.main()}')