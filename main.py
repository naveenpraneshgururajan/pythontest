def checkPalindrome(s):
    #To lowercase everything
    s=s.lower()
    # print(s.toLower())
    #Remove the Spaces
    s=s.replace(' ','')

    if(len(s)%2!=0):
        afterReversing = s[::-1]
        if(s == afterReversing):
            return True
        else:
            return False
        
    else:
        print("its not a palindrome")


print(checkPalindrome("Ma m"))
   