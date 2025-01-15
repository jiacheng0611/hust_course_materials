package homework.ch11_13.p4;

public class CompositeComponent extends Component{
    protected ComponentList children = new ComponentList();
    public CompositeComponent() {

    }
    public CompositeComponent(int id,String name,double price) {
        super(id,name,price);
    }
    public void add(Component component) throws UnsupportedOperationException{
        children.add(component);
    }

    @Override
    public double calcPrice() {
        double price = 0;
        for(Component component : this.children){
            price += component.getPrice();
        }
        return price;
    }

    public ComponentIterator createIterator(){
        return new CompositeIterator(children);
    }

    public void remove(Component component) throws UnsupportedOperationException{
        children.remove(component);
        this.price -= component.getPrice();
    }
    public String toString(){
        String s=super.toString();
        String ss="";
        for(Component c : children){
            ss+=c.toString();
        }
        return s+"sub-components of "+this.getName()+":\n"+ss;
    }
}
