if __name__ == '__main__':
    s = input()
    a = s.split('.')
    if a[-1].lower() == 'py': print('yes')
    else: print('no')