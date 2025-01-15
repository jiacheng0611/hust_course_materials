package homework.ch11_13.p4;

import java.util.ArrayList;

public class ComponentList extends ArrayList<Component> implements ComponentIterator{
    private int position = -1;
    public ComponentList(){

    }
    public ComponentIterator createIterator(){
        return new NullIterator();
    }
    public boolean hasNext(){
        return position+1 < this.size();
    }
    public Component next(){
        if(hasNext()){
            position++;
            return this.get(position);
        }
        else {
            return null;
        }
    }
}
