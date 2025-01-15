package homework.ch11_13.p4;

import java.util.ArrayList;
import java.util.List;

public class CompositeIterator implements ComponentIterator{
    protected List<ComponentIterator> iterators=new ArrayList<>();

    public CompositeIterator(ComponentIterator Componentiterator) {
        iterators.add(Componentiterator);
    }
    public boolean hasNext() {
        if(iterators.isEmpty()){
            return false;
        }
        ComponentIterator iterator = iterators.get(iterators.size() - 1);
        if(!iterator.hasNext()){
            iterators.remove(iterators.size() - 1);
            return hasNext();
        }
        else{
            return true;
        }
    }
    public Component next() {
        if(hasNext()){
            ComponentIterator iterator = iterators.get(iterators.size() - 1);
            Component component = (Component)iterator.next();
            iterators.add(component.createIterator());
            return component;
        }
        return null;
    }
}
