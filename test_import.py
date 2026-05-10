import sys
sys.path.insert(0, r"c:\Users\fbrev\Documents\Acadêmico\Simulações\Python\GCN+LNPCC")
try:
    import lnpcc
    print("LOADED LNPCC FROM:", getattr(lnpcc, "__file__", "UNKNOWN"))
except Exception as e:
    import traceback
    traceback.print_exc()
