import py_compile,sys,os
errors=[]
for root,dirs,files in os.walk(r'.'):
    for f in files:
        if f.endswith('.py'):
            path=os.path.join(root,f)
            try:
                py_compile.compile(path,doraise=True)
            except Exception as e:
                errors.append((path,str(e)))
if errors:
    print('FAILED')
    for p,e in errors:
        print(p)
        print(e)
    sys.exit(2)
print('OK')
