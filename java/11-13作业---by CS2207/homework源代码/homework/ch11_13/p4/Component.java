package homework.ch11_13.p4;

public abstract class Component extends Object {
    protected int id;
    protected String name;
    protected double price;
    public Component() {

    }
    public Component(int id, String name, double price) {
        this.id = id;
        this.name = name;
        this.price = price;
    }
    public int getId(){
        return id;
    }
    public String getName(){
        return name;
    }
    public double getPrice(){
        this.price=calcPrice();
        return price;
    }
    public void setId(int id){
        this.id = id;
    }
    public void setName(String name){
        this.name = name;
    }
    public void setPrice(double price){
        this.price = price;
    }

    public abstract void add(Component component);
    public abstract double calcPrice();
    public abstract ComponentIterator createIterator();
    public abstract void remove(Component component);

    @Override
    public String toString() {
        return "id: " + this.getId() + ", name: " + this.getName() + ", price: " + this.getPrice() + "\n";
    }

    @Override
    public boolean equals(Object obj){
        if(obj instanceof Component){
            if(this.getId()==((Component)obj).getId()){
                return true;
            }
        }
        return false;
    }
}
