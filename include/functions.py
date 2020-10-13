def processInputTarget(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text
