for i in range(9):
    if i>0:print()
    for j in range(i+1):
        print(str(j+1) +'x'+str(i+1)+'=' + str((i+1)*(j+1)), end =' ')