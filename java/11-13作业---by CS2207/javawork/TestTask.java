import java.util.ArrayList;

interface Task {
    /**
     * 执行具体任务的接口方法
     */
    public abstract void execute();
}
interface TaskService {
    /**
     * 执行任务接口列表中的每个任务
     */
    public void executeTasks();
    /**
     * 添加任务
     * @param t 新添加的任务
     */
    public void addTask(Task t);
}
class task1 implements Task{

    public void execute(){
        System.out.println("task1 executed");
    }
}
class task2 implements Task{

    public void execute(){
        System.out.println("task2 executed");
    }
}
class task3 implements Task{

    public void execute(){
        System.out.println("task3 executed");
    }
}
class TaskServiceImpl implements TaskService{
    ArrayList<Task> tasks;
    public TaskServiceImpl() {
        tasks = new ArrayList<>();
    }
    public void executeTasks(){
        for(Task t : tasks){
            t.execute();
        }
    }
    public void addTask(Task t){
        tasks.add(t);
    }
}
public class TestTask {
    public static void main(String[] args) {
        Task task1 = new task1();
        Task task2 = new task2();
        Task task3 = new task3();

        TaskServiceImpl taskService = new TaskServiceImpl();

        taskService.addTask(task1);
        taskService.addTask(task2);
        taskService.addTask(task3);
        taskService.executeTasks();
    }
}
