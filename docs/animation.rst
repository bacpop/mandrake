Creating animations
===================
You can also create an animation of the embedding process as it runs simply
by adding the ``--animate`` flag. This works with both the CPU and GPU versions
of the algorithm, and will output a video file ``mandrake.embedding_animation.mp4``:

.. image:: images/mandrake.embedding_animation.gif
   :alt:  Animation of an embedding
   :align: center

The top panel shows the embedding, the bottom panels shows the iteration and :math:`Eq`
at the iteration. The learning rate always decreases linearly, so it is not plotted.

You will see two further progress bars after plotting::

    Creating animation
    100%|█████████████████████████████████████████████████████| 400/400 [00:11<00:00, 35.73frames/s]
    Saving frame 400 of 400

The first is saving static images of each frame, the second is encoding these into
a video using ``ffmpeg``.

Details
-------
- The colours are the 'final' colours of HDBSCAN run on the embedding result,
  or the provided labels. Black points are noise points.
- The dimensions are rescaled to have unit standard deviation in each direction
  at every frame.
- Animations have 400 frames played at 20fps, resulting in a 20s animation.
- Resolution is 1920x2560px.
- Samples are taken more regularly at the start, when learning is happening at a
  greater rate and points move more, and less frequently at the end, when learning is
  slow and points move less. Specifically, samples are taken at a rate such that the
  total amount of learning is divided equally, which is proportional to :math:`1 - \sqrt{1-x}`.