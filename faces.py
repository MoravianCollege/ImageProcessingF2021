def yield_for_change(widget, attribute):
    """
    Pause a generator to wait for a widget change event.

    This is a decorator for a generator function which pauses the generator on yield
    until the given widget attribute changes. The new value of the attribute is
    sent to the generator and is the value of the yield.
    """
    from functools import wraps
    def f(iterator):
        @wraps(iterator)
        def inner():
            i = iterator()
            def next_i(change):
                try:
                    i.send(change.new)
                except StopIteration as e:
                    widget.unobserve(next_i, attribute)
            widget.observe(next_i, attribute)
            # start the generator
            next(i)
        return inner
    return f

def add_with_prompt(recognizer, im):
    """Add an image to the recognizer after prompting the user for the name."""
    import ipywidgets as widgets
    from IPython.display import Image, Pretty, display, clear_output
    from cv2 import imencode
    name,confidence = recognizer.recognize(im)
    clear_output()
    display(Image(imencode('.jpg', im,)[1].tostring(), width=im.shape[1], height=im.shape[0]))
    txt = widgets.Text(placeholder='Who is this?', continuous_update=False)
    @yield_for_change(txt, 'value')
    def f():
        name = yield
        recognizer.add(name, im)
        clear_output()
    f()
    display(txt)
    display(Pretty(f'We think it might be {name} (distance = {round(confidence)})'))
