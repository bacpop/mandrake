Web app
=======
You can find a version of mandrake at https://gtonkinhill.github.io/mandrake-web/

This is a static web app which uses a WebAssembly version of mandrake to run
in your browser. This means no sequence data is transmitted across the network
and the analysis is run entirely locally in your browser, which:

- Means data does not have to be uploaded, which can take a while.
- We see and store no sequence, so your data never leaves your machine, or is
  ever even read by us at all.
- We don't have to maintain a server (which is good for us), which is typically
  more reliable (which good for you).

The use of this is hopefully fairly self-explanatory:

1. Choose an input sequence file.
2. Select the type of distance based on the type of the input file
   (SNP distances, accessory or sketch; see :doc:`input`).
3. (optionally) set labels to colour the output with.
4. (optionally) change the SCE algorithm parameters.

Performance
-----------
The web app is suitable for up to a few thousand sequences. If you are analysing
more sequences than that you should use the command line tool.

The web app can only use a single CPU core. It is usually faster in 'native' browers,
Safari on OS X and Edge on Windows.
