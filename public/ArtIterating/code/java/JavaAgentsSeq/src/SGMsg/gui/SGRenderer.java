package SGMsg.gui;

import SGMsg.agent.SGAgent;

import javax.swing.*;
import java.awt.*;
import java.awt.geom.Rectangle2D;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by jonathan on 05/12/16.
 */
public class SGRenderer extends JPanel {
    private List<SGAgent> as;

    private final static int BORDER_X = 10;
    private final static int BORDER_Y = 10;

    private final static Color BLUE = new Color(0.0f,0.0f, 0.7f);
    private final static Color GREEN = new Color(0.0f,0.55f, 0.0f);
    private final static Color YELLOW = new Color(1.0f,0.7f, 0.0f);
    private final static Color RED = new Color(0.7f,0.0f, 0.0f);

    private int columns;
    private int rows;

    public SGRenderer(int columns, int rows) {
        this.columns = columns;
        this.rows = rows;
        this.as = new ArrayList<>();
    }

    @Override
    public void paint(Graphics g) {
        Graphics2D g2 = (Graphics2D) g;

        g2.setColor(Color.WHITE);
        g2.fillRect(0, 0, this.getWidth(), this.getHeight());

        double width = this.getWidth() - 2 * BORDER_X;
        double height = this.getHeight() - 2 * BORDER_Y;

        double cellWidth = width / columns;
        double cellHeight = height / rows;

        for (SGAgent a : this.as) {
            double x = BORDER_X + (a.getCell().getX() * cellWidth);
            double y = BORDER_Y + (a.getCell().getY() * cellHeight);

            if ( SGAgent.SGState.Cooperator == a.getPrevState() && SGAgent.SGState.Cooperator == a.getCurrState() ) {
                g2.setColor( BLUE );

            } else if ( SGAgent.SGState.Defector == a.getPrevState() && SGAgent.SGState.Defector == a.getCurrState() ) {
                g2.setColor( RED );

            } else if ( SGAgent.SGState.Defector == a.getPrevState() && SGAgent.SGState.Cooperator == a.getCurrState() ) {
                g2.setColor( GREEN );

            } else if ( SGAgent.SGState.Cooperator == a.getPrevState() && SGAgent.SGState.Defector == a.getCurrState() ) {
                g2.setColor( YELLOW );

            }

            Shape r = new Rectangle2D.Double(x, y, cellWidth, cellHeight);
            g2.fill(r);
        }
    }

    public void render(List<SGAgent> as) {
        this.as = as;
        this.repaint();
    }
}
