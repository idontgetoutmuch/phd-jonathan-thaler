package SGMsg.gui;

import SGMsg.agent.SGAgent;
import agent.ISimulationObserver;

import javax.swing.*;
import java.awt.*;
import java.util.LinkedHashMap;
import java.util.List;

/**
 * Created by jonathan on 05/12/16.
 */
public class SGFrontend extends JFrame implements ISimulationObserver<SGAgent> {
    private SGRenderer renderer;

    public SGFrontend(int cols, int rows) {
        super("Spatial Game Par");

        this.renderer = new SGRenderer(cols, rows);

        this.setDefaultCloseOperation( EXIT_ON_CLOSE );
        this.setContentPane( renderer );
        this.setSize( new Dimension( 1000, 1000 ));
        this.setVisible(true);
    }

    @Override
    public boolean simulationStep(LinkedHashMap<Integer, SGAgent> as) {
        this.renderer.render(as);

        try {
            Thread.sleep( 30 );
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        return true;
    }
}
