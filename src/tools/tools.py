def content_merge(row):
    text = f'''Assessment: {row["Assessment"]}\n
           Subjective Section: {row["Subjective Sections"]}\n
           Objective Section: {row["Objective Sections"]}\n'''
    return text