主要的update：处理了A* + VFH算法，以及优化了gdino导航的逻辑。
具体实现参考action_processor.py，代码有点长，但是逻辑应该还好。实现模式和我们早上描述的是一样的。（除了没有管VLM的事）
具体接口：

首先使用的action config是new_action_format.json. 这个文件的格式相对还算比较简单，需要设置位置，target name和读取的mask位置。

还有一些和config相关的事情，就是在action_processor前面的部分有一个 NavigationConfig，这里面有一些和导航相关的参数。其中，min_search_radius， max_search_radius是用来处理我们从障碍物到距离最近的可行走点/未知点的转换到，min_search_radius保证距离最近的可行走点/未知点和原本点至少有20像素，从而避免只调整很小格子的情况（几次跑下来发现会有只调整了2pixel的情况）。我在测试的时候目前还是用的像素距离，也许可以调整成m作为单位（从而保证统一？）但其实我试了一下，就算把这个值设置成0好像也不会导致什么区别。当我写下刚才这部分话的时候已经把这个参数给删掉了。（（

我的想法是也许可以把这些参数整合到一开始的config中，但是我不是很确定怎么处理这个事。

我顺便把histogram拿掉了。

如果希望运行main函数，可以直接按照我现在的设置，我把用到的wall mask发给你（我记得原本文件夹里没有）。
